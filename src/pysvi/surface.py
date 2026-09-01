# src/pysvi/surface.py
"""Fitted volatility surface: evaluation, interpolation, and pricing.

`VolSurface` turns per-slice calibration results into the object quant
work actually consumes: model -> calibration -> fitted surface. It owns
calibrated slices across maturities and exposes vectorized evaluation
(IVs, total variance, ATM level/skew/curvature), maturity interpolation
between slices, arbitrage verification, and a Black-76 pricing and
Greeks layer on the slice forwards.

`calibrate_surface` is the calendar-aware fitter: it orders expiries,
derives per-slice inputs, chains the prior slice's total variance into
each NO_CALENDAR penalty automatically, and — for eSSVI — fits the
global term structure jointly across all slices.

Conventions
-----------
* Pricing is Black-76 on the slice forward with a flat continuously
  compounded rate ``r`` (default 0): ``C = e^{-rT}[F N(d1) - K N(d2)]``.
* Greeks hold the implied volatility fixed (sticky-strike): delta and
  gamma are with respect to the forward, vega is per unit volatility,
  theta is per year of calendar time.
* Between fitted maturities the surface interpolates (linearly in total
  variance at fixed log-moneyness by default; see ``interp_method``);
  forwards interpolate log-linearly. Extrapolation beyond the fitted
  maturity range raises.
"""

from bisect import bisect_left
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
from loguru import logger
from scipy.special import ndtr

from . import _kernels
from .models import (
    ArbitrageFreedom, DirectSVI, ESSVI, JumpWings, Parametrization, SABR, SSVI,
    _initialization, _minimize_with_starts, _multistart_variants, _penalty_grid,
    _prepare_loss_inputs, essvi_total_variance,
)
from .calibration import calibrate_slice, get_model, prepare_slice
from .diagnostics import ArbitrageReport, check_arbitrage

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_INTERP_METHODS = ("total_variance", "theta")


def _npdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _is_call(cp: str) -> bool:
    flag = str(cp).lower()
    if flag in ("call", "c"):
        return True
    if flag in ("put", "p"):
        return False
    raise ValueError(f"cp must be 'call' or 'put', got {cp!r}")


def _shape_like(values, original):
    """Return a float for scalar input, the array otherwise."""
    return float(values[0]) if np.ndim(original) == 0 else values


def _auto_slice_kwargs(instance, T, df_slice, model_kwargs, theta_by_T, theta_ref):
    """Derive the per-slice calibrate kwargs for a model instance."""
    kwargs = dict(model_kwargs)
    if isinstance(instance, ESSVI):
        kwargs["theta"] = theta_by_T[T]
        kwargs.setdefault("theta_ref", theta_ref)
    elif isinstance(instance, SSVI):
        kwargs["theta"] = theta_by_T[T]
    elif isinstance(instance, JumpWings):
        kwargs["T"] = T
    elif isinstance(instance, SABR):
        kwargs["T"] = T
        kwargs["F"] = float(df_slice["implied_forward"].iloc[0])
        kwargs.setdefault("beta", 0.5)
    return kwargs


def _data_thetas(instance, groups):
    """Per-slice ATM total variance for SSVI/eSSVI, plus the median ref."""
    theta_by_T: Dict[float, float] = {}
    theta_ref = None
    if isinstance(instance, (SSVI, ESSVI)):
        for T, g in groups:
            theta_by_T[T] = float(np.nanmin(g["iv"] ** 2 * g["maturity"]))
        theta_ref = float(np.median(list(theta_by_T.values())))
    return theta_by_T, theta_ref


class VolSurface:
    """A fitted implied-volatility surface across maturities.

    Construct via :meth:`fit` (independent per-slice calibration),
    :func:`calibrate_surface` (calendar-aware), or directly from
    calibrated slices::

        surface = VolSurface.fit(df, model="svi")
        surface = VolSurface(model, {0.25: params_1, 0.5: params_2})

    Direct construction requires each params dict to carry ``'forward'``
    (as returned by ``calibrate_slice``).

    Parameters
    ----------
    model : Parametrization
        The model instance all slices were calibrated with.
    slices : mapping or iterable of (maturity, params)
        Calibrated slices; sorted by maturity internally.
    r : float, default 0.0
        Flat continuously compounded discount rate used by the pricing
        layer.
    interp_method : str, default "total_variance"
        Maturity interpolation between fitted slices. "total_variance"
        blends w(k) linearly in T at fixed log-moneyness (model-agnostic;
        calendar-free whenever the bracketing slices are). "theta"
        (SSVI/eSSVI only) interpolates the ATM total variance and shape
        parameters, yielding a genuine parametric slice at any maturity.
    """

    def __init__(
        self,
        model: Parametrization,
        slices: Union[
            Mapping[float, Dict[str, float]],
            Iterable[Tuple[float, Dict[str, float]]],
        ],
        r: float = 0.0,
        interp_method: str = "total_variance",
    ) -> None:
        if isinstance(slices, Mapping):
            slices = slices.items()
        ordered = sorted((float(T), dict(params)) for T, params in slices)
        if not ordered:
            raise ValueError("VolSurface requires at least one calibrated slice")
        maturities = [T for T, _ in ordered]
        if len(set(maturities)) != len(maturities):
            raise ValueError(f"duplicate maturities in slices: {maturities}")
        for T, params in ordered:
            if T <= 0:
                raise ValueError(f"maturities must be positive, got {T}")
            forward = params.get("forward")
            if forward is None or not np.isfinite(forward) or forward <= 0:
                raise ValueError(
                    f"slice T={T:g} lacks a positive 'forward' entry; "
                    "calibrate_slice adds it, direct construction must supply it"
                )
        if interp_method not in _INTERP_METHODS:
            raise ValueError(
                f"unknown interp_method {interp_method!r}; choose from {_INTERP_METHODS}"
            )
        if interp_method == "theta" and not isinstance(model, (SSVI, ESSVI)):
            raise ValueError(
                "interp_method='theta' requires an SSVI or eSSVI model"
            )
        self.model = model
        self.r = float(r)
        self.interp_method = interp_method
        self._slices = ordered

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        df,
        model: Union[str, Parametrization] = "svi",
        arbitrage_condition: ArbitrageFreedom = ArbitrageFreedom.QUASI,
        r: float = 0.0,
        interp_method: str = "total_variance",
        **model_kwargs,
    ) -> "VolSurface":
        """Calibrate every maturity slice of an option panel independently.

        Expects the ``calibrate_slice`` schema: columns ``strike``,
        ``iv``, ``maturity``, ``implied_forward``, with multiple
        maturities in one DataFrame. Model-specific per-slice arguments
        are derived automatically: ``theta`` (ATM total variance) for
        SSVI/eSSVI with ``theta_ref`` defaulting to the median across
        slices, ``T`` for jump-wings, and ``T``/``F`` for SABR (``beta``
        defaults to 0.5 — override via ``model_kwargs``).

        Calibration controls (``objective``, ``loss``, ``f_scale``,
        ``initialization``) pass through to every slice via
        ``model_kwargs``. Slices that fail to calibrate are skipped with
        a warning; fitting fails only if no slice succeeds. For
        cross-slice calendar enforcement use :func:`calibrate_surface`.

        Parameters
        ----------
        df : pd.DataFrame
            Multi-expiry option panel.
        model : str or Parametrization, default "svi"
            Factory name or a model instance.
        arbitrage_condition : ArbitrageFreedom, default QUASI
            Used when ``model`` is a factory name.
        r : float, default 0.0
            Flat discount rate for the pricing layer.
        interp_method : str, default "total_variance"
            Maturity interpolation method (see the class docstring).
        **model_kwargs
            Forwarded to every per-slice calibration.

        Returns
        -------
        VolSurface
        """
        instance = (
            get_model(model, arbitrage_condition)
            if isinstance(model, str) else model
        )
        groups = sorted(
            ((float(T), g) for T, g in df.groupby("maturity")),
            key=lambda item: item[0],
        )
        if not groups:
            raise ValueError("VolSurface.fit: empty input panel")

        theta_by_T, theta_ref = _data_thetas(instance, groups)
        slices = []
        for T, g in groups:
            kwargs = _auto_slice_kwargs(
                instance, T, g, model_kwargs, theta_by_T, theta_ref
            )
            params = calibrate_slice(g, instance, **kwargs)
            if params is None:
                logger.warning(
                    f"VolSurface.fit: slice T={T:g} failed to calibrate; skipping"
                )
                continue
            slices.append((T, params))
        if not slices:
            raise ValueError("VolSurface.fit: no slice calibrated successfully")
        return cls(instance, slices, r=r, interp_method=interp_method)

    # ── Slice access and maturity location ───────────────────────────

    @property
    def maturities(self) -> np.ndarray:
        """Fitted maturities, ascending."""
        return np.array([T for T, _ in self._slices])

    def _locate(self, maturity):
        """Locate a maturity: ("exact", T, params) or ("interp", lo, hi, lam)."""
        T = float(maturity)
        for Ti, params in self._slices:
            if abs(Ti - T) <= 1e-12 * max(1.0, abs(T)):
                return ("exact", Ti, params)
        Ts = [Ti for Ti, _ in self._slices]
        if T < Ts[0] or T > Ts[-1]:
            fitted = ", ".join(f"{Ti:g}" for Ti in Ts)
            raise ValueError(
                f"maturity {T:g} is outside the fitted range [{Ts[0]:g}, {Ts[-1]:g}]"
                f" (fitted: {fitted}); extrapolation is not supported"
            )
        j = bisect_left(Ts, T)
        T_lo, p_lo = self._slices[j - 1]
        T_hi, p_hi = self._slices[j]
        lam = (T - T_lo) / (T_hi - T_lo)
        return ("interp", (T_lo, p_lo), (T_hi, p_hi), lam)

    def _slice(self, maturity) -> Tuple[float, Dict[str, float]]:
        loc = self._locate(maturity)
        if loc[0] != "exact":
            fitted = ", ".join(f"{Ti:g}" for Ti, _ in self._slices)
            raise ValueError(
                f"maturity {float(maturity):g} is not a fitted slice"
                f" (fitted: {fitted})"
            )
        return loc[1], loc[2]

    def _theta_params(self, lo, hi, lam) -> Dict[str, float]:
        """Synthetic SSVI/eSSVI params at an interpolated maturity."""
        (T_lo, p_lo), (T_hi, p_hi) = lo, hi
        blend = {
            key: (1.0 - lam) * p_lo[key] + lam * p_hi[key]
            for key in p_lo
            if key != "forward" and key in p_hi
        }
        blend["forward"] = float(np.exp(
            (1.0 - lam) * np.log(p_lo["forward"])
            + lam * np.log(p_hi["forward"])
        ))
        if "rho_theta" in blend and isinstance(self.model, ESSVI):
            # keep the stored rho(theta) consistent with the blended theta
            blend["rho_theta"] = self.model._rho_of(
                blend["theta"], blend["theta_ref"],
                blend["rho0"], blend["rho1"], blend["alpha"],
            )
        return blend

    def slice_at(self, maturity) -> Dict[str, float]:
        """Parameter dict of the (possibly synthetic) slice at ``maturity``.

        Exact for fitted maturities. Between slices this requires a
        parametric interpolation method (``interp_method="theta"``);
        the default total-variance blend has no parameter representation
        and raises here (evaluation methods still work at any maturity).
        """
        loc = self._locate(maturity)
        if loc[0] == "exact":
            return dict(loc[2])
        if self.interp_method != "theta":
            raise ValueError(
                "slice_at between fitted maturities requires "
                "interp_method='theta' (the total-variance blend has no "
                "parameter representation); evaluation methods such as "
                "iv/total_variance/price work at any maturity in range"
            )
        return self._theta_params(loc[1], loc[2], loc[3])

    def params(self, maturity) -> Dict[str, float]:
        """Calibrated parameter dict of the fitted slice at ``maturity`` (a copy)."""
        return dict(self._slice(maturity)[1])

    def forward(self, maturity) -> float:
        """Forward price at ``maturity`` (log-linear between fitted slices)."""
        loc = self._locate(maturity)
        if loc[0] == "exact":
            return float(loc[2]["forward"])
        (_, p_lo), (_, p_hi), lam = loc[1], loc[2], loc[3]
        return float(np.exp(
            (1.0 - lam) * np.log(p_lo["forward"]) + lam * np.log(p_hi["forward"])
        ))

    # ── Evaluation ───────────────────────────────────────────────────

    def _w_at(self, k: np.ndarray, maturity) -> np.ndarray:
        loc = self._locate(maturity)
        if loc[0] == "exact":
            return self.model.total_variance(k, loc[2])
        lo, hi, lam = loc[1], loc[2], loc[3]
        if self.interp_method == "theta":
            return self.model.total_variance(k, self._theta_params(lo, hi, lam))
        w_lo = self.model.total_variance(k, lo[1])
        w_hi = self.model.total_variance(k, hi[1])
        return (1.0 - lam) * w_lo + lam * w_hi

    def _dw_at(self, k: np.ndarray, maturity, second: bool = False) -> np.ndarray:
        deriv = self.model.d2w_dk2 if second else self.model.dw_dk
        loc = self._locate(maturity)
        if loc[0] == "exact":
            return deriv(k, loc[2])
        lo, hi, lam = loc[1], loc[2], loc[3]
        if self.interp_method == "theta":
            return deriv(k, self._theta_params(lo, hi, lam))
        # derivative of the linear blend is the blend of derivatives
        return (1.0 - lam) * deriv(k, lo[1]) + lam * deriv(k, hi[1])

    def total_variance(self, k, maturity):
        """Total variance w(k) at any maturity in the fitted range."""
        values = self._w_at(np.atleast_1d(np.asarray(k, dtype=np.float64)), maturity)
        return _shape_like(values, k)

    def iv(self, strike, maturity):
        """Implied volatility at absolute strike(s), any maturity in range."""
        T = float(maturity)
        F = self.forward(T)
        K = np.atleast_1d(np.asarray(strike, dtype=np.float64))
        if np.any(K <= 0):
            raise ValueError("strikes must be positive")
        k = np.log(K / F)
        w = self._w_at(k, T)
        sigma = np.sqrt(np.maximum(w, 0.0) / T)
        return _shape_like(sigma, strike)

    def atm_vol(self, maturity) -> float:
        """At-the-money (k = 0) implied volatility."""
        T = float(maturity)
        w0 = float(self._w_at(np.array([0.0]), T)[0])
        return float(np.sqrt(max(w0, 0.0) / T))

    def skew(self, maturity) -> float:
        """ATM total-variance skew dw/dk at k = 0."""
        return float(self._dw_at(np.array([0.0]), maturity)[0])

    def curvature(self, maturity) -> float:
        """ATM total-variance curvature d2w/dk2 at k = 0."""
        return float(self._dw_at(np.array([0.0]), maturity, second=True)[0])

    # ── Diagnostics ──────────────────────────────────────────────────

    def check_arbitrage(self, **kwargs) -> ArbitrageReport:
        """Run the arbitrage diagnostics over all fitted slices.

        Forwards to :func:`pysvi.diagnostics.check_arbitrage` (butterfly,
        Lee wing bounds, calendar); keyword arguments (``k_min``,
        ``k_max``, ``n_grid``, ``tol``, ``k_data``) pass through.
        """
        return check_arbitrage(self.model, self._slices, **kwargs)

    # ── Black-76 pricing and Greeks ──────────────────────────────────

    def _black_inputs(self, strike, maturity):
        T = float(maturity)
        F = self.forward(T)
        K = np.atleast_1d(np.asarray(strike, dtype=np.float64))
        if np.any(K <= 0):
            raise ValueError("strikes must be positive")
        k = np.log(K / F)
        w = np.maximum(self._w_at(k, T), 1e-16)
        sqrt_w = np.sqrt(w)
        d1 = (-k + 0.5 * w) / sqrt_w
        d2 = d1 - sqrt_w
        return T, F, K, sqrt_w, d1, d2

    def price(self, strike, maturity, cp: str = "call"):
        """Black-76 option price at absolute strike(s).

        ::

            call = e^{-rT} [F N(d1) - K N(d2)]
            put  = e^{-rT} [K N(-d2) - F N(-d1)]

        with d1 = (log(F/K) + w/2)/sqrt(w), d2 = d1 - sqrt(w), and w the
        surface total variance at the strike.
        """
        is_call = _is_call(cp)
        T, F, K, sqrt_w, d1, d2 = self._black_inputs(strike, maturity)
        disc = np.exp(-self.r * T)
        if is_call:
            value = disc * (F * ndtr(d1) - K * ndtr(d2))
        else:
            value = disc * (K * ndtr(-d2) - F * ndtr(-d1))
        return _shape_like(value, strike)

    def delta(self, strike, maturity, cp: str = "call"):
        """Black-76 forward delta, e^{-rT} N(d1) (call) or -e^{-rT} N(-d1) (put).

        Holds the implied volatility fixed (sticky-strike).
        """
        is_call = _is_call(cp)
        T, F, K, sqrt_w, d1, _ = self._black_inputs(strike, maturity)
        disc = np.exp(-self.r * T)
        value = disc * ndtr(d1) if is_call else -disc * ndtr(-d1)
        return _shape_like(value, strike)

    def gamma(self, strike, maturity):
        """Black-76 gamma, e^{-rT} phi(d1) / (F sqrt(w)). Same for calls and puts."""
        T, F, K, sqrt_w, d1, _ = self._black_inputs(strike, maturity)
        value = np.exp(-self.r * T) * _npdf(d1) / (F * sqrt_w)
        return _shape_like(value, strike)

    def vega(self, strike, maturity):
        """Black-76 vega per unit volatility, e^{-rT} F phi(d1) sqrt(T)."""
        T, F, K, sqrt_w, d1, _ = self._black_inputs(strike, maturity)
        value = np.exp(-self.r * T) * F * _npdf(d1) * np.sqrt(T)
        return _shape_like(value, strike)

    def theta(self, strike, maturity, cp: str = "call"):
        """Black-76 theta per year of calendar time, holding sigma fixed.

        ::

            theta = r V - e^{-rT} F phi(d1) sigma / (2 sqrt(T))
        """
        T, F, K, sqrt_w, d1, _ = self._black_inputs(strike, maturity)
        decay = np.exp(-self.r * T) * F * _npdf(d1) * sqrt_w / (2.0 * T)
        value = self.r * np.atleast_1d(self.price(strike, maturity, cp)) - decay
        return _shape_like(value, strike)


# ── Calendar-aware surface calibration ───────────────────────────────


def _warn_ssvi_admissibility(slices) -> None:
    """Gatheral-Jacquier sufficient no-butterfly bounds for SSVI-form slices.

    One aggregated warning when theta*phi*(1+abs(rho)) > 4 or
    theta*phi^2*(1+abs(rho)) > 4 (phi = eta/sqrt(theta)) on any slice.
    The bounds are sufficient, not necessary — violating them does not
    prove arbitrage, it only leaves the slice outside the proven-safe
    region, so verify with check_arbitrage().
    """
    outside = []
    for T, params in slices:
        theta = params["theta"]
        rho = params.get("rho_theta", params.get("rho"))
        phi = params["eta"] / np.sqrt(theta)
        lvl = theta * phi * (1.0 + abs(rho))
        crv = theta * phi * phi * (1.0 + abs(rho))
        if lvl > 4.0 or crv > 4.0:
            outside.append(f"T={T:g}")
    if outside:
        logger.warning(
            "SSVI admissibility: slices outside the Gatheral-Jacquier "
            f"sufficient no-butterfly bounds: {', '.join(outside)} "
            "(the bounds are conservative; verify with check_arbitrage())"
        )


def calibrate_surface(
    df,
    model: Union[str, Parametrization] = "ssvi",
    enforce_calendar: bool = True,
    arbitrage_condition: ArbitrageFreedom = ArbitrageFreedom.QUASI,
    r: float = 0.0,
    interp_method: str = "total_variance",
    **model_kwargs,
) -> VolSurface:
    """Calendar-aware multi-expiry calibration returning a VolSurface.

    Orders the expiries and calibrates them oldest-first, automatically
    evaluating each fitted slice on the next slice's penalty grid and
    passing it as ``w_prev`` — the manual chaining the per-slice API
    requires. With ``enforce_calendar`` the NO_CALENDAR flag is added to
    the model's arbitrage condition, and for SSVI/eSSVI the per-slice
    ATM total variances are made non-decreasing before fitting.

    For eSSVI the global term structure (rho0, rho1, alpha, eta) is
    fitted **jointly across all slices** against the shared theta_ref
    (median ATM total variance), rather than independently per slice —
    every returned slice carries identical shape parameters.

    After fitting, SSVI-form slices are checked against the
    Gatheral-Jacquier sufficient no-butterfly bounds and a warning is
    logged for slices outside the proven-safe region.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-expiry option panel (``calibrate_slice`` schema).
    model : str or Parametrization, default "ssvi"
        Factory name or model instance. DirectSVI is rejected when
        ``enforce_calendar`` is set (no penalty support).
    enforce_calendar : bool, default True
        Add NO_CALENDAR to the arbitrage condition and chain w_prev.
    arbitrage_condition : ArbitrageFreedom, default QUASI
        Base condition when ``model`` is a factory name (NO_BUTTERFLY
        may be OR-ed in; NO_CALENDAR is added by ``enforce_calendar``).
    r : float, default 0.0
        Flat discount rate for the pricing layer.
    interp_method : str, default "total_variance"
        Maturity interpolation method for the returned surface.
    **model_kwargs
        Calibration controls and model extras, forwarded to every slice
        (``beta`` for SABR, etc.). The bid_ask objective is not
        supported in the joint eSSVI path.

    Returns
    -------
    VolSurface
    """
    if isinstance(model, str):
        condition = arbitrage_condition
        if enforce_calendar:
            condition |= ArbitrageFreedom.NO_CALENDAR
        instance = get_model(model, condition)
    else:
        instance = model
        if enforce_calendar and ArbitrageFreedom.NO_CALENDAR not in instance.arbitrage_condition:
            instance = type(instance)(
                arbitrage_condition=instance.arbitrage_condition
                | ArbitrageFreedom.NO_CALENDAR
            )
    if enforce_calendar and isinstance(instance, DirectSVI):
        raise ValueError(
            "DirectSVI does not support penalty-based calendar enforcement; "
            "use enforce_calendar=False or an iterative model"
        )

    groups = sorted(
        ((float(T), g) for T, g in df.groupby("maturity")),
        key=lambda item: item[0],
    )
    if not groups:
        raise ValueError("calibrate_surface: empty input panel")

    prepared = []
    for T, g in groups:
        k_i, w_i, F_i = prepare_slice(g)
        if k_i is None:
            logger.warning(
                f"calibrate_surface: slice T={T:g} has insufficient data; skipping"
            )
            continue
        prepared.append((T, g, k_i, w_i, F_i))
    if not prepared:
        raise ValueError("calibrate_surface: no usable slice in the panel")

    theta_by_T, theta_ref = _data_thetas(instance, (item[:2] for item in prepared))
    if enforce_calendar and theta_by_T:
        ordered_T = [item[0] for item in prepared]
        raw = np.array([theta_by_T[T] for T in ordered_T])
        monotone = np.maximum.accumulate(raw)
        if np.any(monotone > raw):
            logger.warning(
                "calibrate_surface: per-slice ATM total variances were not "
                "non-decreasing; clipped upward to enforce a monotone theta(T)"
            )
        theta_by_T = dict(zip(ordered_T, (float(x) for x in monotone)))

    if isinstance(instance, ESSVI):
        slices = _calibrate_essvi_global(
            instance, prepared, theta_by_T, theta_ref, enforce_calendar, model_kwargs
        )
    else:
        slices = []
        prev_params = None
        for T, g, k_i, w_i, F_i in prepared:
            kwargs = _auto_slice_kwargs(
                instance, T, g, model_kwargs, theta_by_T, theta_ref
            )
            if enforce_calendar and prev_params is not None:
                grid = _penalty_grid(k_i)
                kwargs["w_prev"] = instance.total_variance(grid, prev_params)
            params = instance.calibrate(k_i, w_i, **kwargs)
            if params is None:
                logger.warning(
                    f"calibrate_surface: slice T={T:g} failed to calibrate; skipping"
                )
                continue
            params["forward"] = F_i
            slices.append((T, params))
            prev_params = params
    if not slices:
        raise ValueError("calibrate_surface: no slice calibrated successfully")

    if isinstance(instance, (SSVI, ESSVI)):
        _warn_ssvi_admissibility(slices)

    return VolSurface(instance, slices, r=r, interp_method=interp_method)


def _calibrate_essvi_global(
    instance, prepared, theta_by_T, theta_ref, enforce_calendar, model_kwargs
):
    """Joint eSSVI fit: one (rho0, rho1, alpha, eta) across all slices."""
    if model_kwargs.get("objective") == "bid_ask":
        raise ValueError(
            "objective='bid_ask' is not supported in the joint eSSVI fit"
        )
    check_bf = ArbitrageFreedom.NO_BUTTERFLY in instance.arbitrage_condition
    init = _initialization(model_kwargs)

    ctx = []
    for T, g, k_i, w_i, F_i in prepared:
        mode, loss_code, weights, w_lo, w_hi = _prepare_loss_inputs(
            k_i, w_i, model_kwargs
        )
        ctx.append((T, k_i, w_i, F_i, theta_by_T[T], mode, loss_code, weights))
    # one common grid spanning all slices, for butterfly and calendar
    k_lo = min(float(item[1].min()) for item in ctx)
    k_hi = max(float(item[1].max()) for item in ctx)
    common_grid = np.linspace(k_lo - 0.5, k_hi + 0.5, 200)
    empty = np.empty(0)

    f_scale = float(model_kwargs.get("f_scale", 1.0))
    core = _kernels.resolve("essvi_obj")
    theta_ref = float(theta_ref)

    def objective(p):
        p = np.asarray(p, dtype=np.float64)
        total = 0.0
        w_prev = empty
        has_prev = False
        for T, k_i, w_i, F_i, theta_i, mode, loss_code, weights in ctx:
            total += core(
                p, k_i, w_i, theta_i, theta_ref, common_grid, w_prev,
                check_bf, enforce_calendar, has_prev,
                mode, weights, empty, empty, loss_code, f_scale,
            )
            if enforce_calendar:
                rho_t = ESSVI._rho_of(theta_i, theta_ref, p[0], p[1], p[2])
                phi = p[3] / np.sqrt(theta_i)
                w_prev = essvi_total_variance(common_grid, theta_i, rho_t, phi)
                has_prev = True
        return total

    x0 = np.array([0.0, -0.5, 0.5, 1.0])
    bounds = [(-0.999, 0.999), (-2.0, 2.0), (-2.0, 2.0), (1e-8, None)]
    starts = _multistart_variants(x0, 0, 3) if init == "multi_start" else [x0]
    res = _minimize_with_starts(
        objective, starts, bounds,
        lbfgs_options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000},
        nm_options={"maxiter": 2000},
    )
    if res is None:
        raise ValueError("calibrate_surface: joint eSSVI fit failed to converge")
    rho0, rho1, alpha, eta = (float(x) for x in res.x)
    if eta <= 0:
        raise ValueError("calibrate_surface: joint eSSVI fit returned eta <= 0")

    slices = []
    for T, k_i, w_i, F_i, theta_i, mode, loss_code, weights in ctx:
        slices.append((T, {
            "rho0": rho0, "rho1": rho1, "alpha": alpha, "eta": eta,
            "theta": float(theta_i), "theta_ref": theta_ref,
            "rho_theta": ESSVI._rho_of(theta_i, theta_ref, rho0, rho1, alpha),
            "forward": float(F_i),
        }))
    return slices
