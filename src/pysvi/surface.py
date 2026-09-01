# src/pysvi/surface.py
"""Fitted volatility surface: evaluation, diagnostics, and Black-76 pricing.

`VolSurface` turns per-slice calibration results into the object quant
work actually consumes: model -> calibration -> fitted surface. It owns
calibrated slices across maturities and exposes vectorized evaluation
(IVs, total variance, ATM level/skew/curvature), arbitrage verification,
and a Black-76 pricing and Greeks layer on the slice forwards.

Conventions
-----------
* Pricing is Black-76 on the slice forward with a flat continuously
  compounded rate ``r`` (default 0): ``C = e^{-rT}[F N(d1) - K N(d2)]``.
* Greeks hold the implied volatility fixed (sticky-strike): delta and
  gamma are with respect to the forward, vega is per unit volatility,
  theta is per year of calendar time.
* Evaluation is per fitted slice; maturity interpolation between slices
  arrives in a later release.
"""

from typing import Dict, Iterable, Mapping, Tuple, Union

import numpy as np
from loguru import logger
from scipy.special import ndtr

from .models import (
    ArbitrageFreedom, ESSVI, JumpWings, Parametrization, SABR, SSVI,
)
from .calibration import calibrate_slice, get_model
from .diagnostics import ArbitrageReport, check_arbitrage

_SQRT_2PI = np.sqrt(2.0 * np.pi)


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


class VolSurface:
    """A fitted implied-volatility surface across maturities.

    Construct via :meth:`fit` (from an option panel) or directly from
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
    """

    def __init__(
        self,
        model: Parametrization,
        slices: Union[
            Mapping[float, Dict[str, float]],
            Iterable[Tuple[float, Dict[str, float]]],
        ],
        r: float = 0.0,
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
        self.model = model
        self.r = float(r)
        self._slices = ordered

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        df,
        model: Union[str, Parametrization] = "svi",
        arbitrage_condition: ArbitrageFreedom = ArbitrageFreedom.QUASI,
        r: float = 0.0,
        **model_kwargs,
    ) -> "VolSurface":
        """Calibrate every maturity slice of an option panel.

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
        a warning; fitting fails only if no slice succeeds. Note that
        NO_CALENDAR's cross-slice chaining is not performed here (a
        dedicated surface calibrator arrives in a later release) —
        verify with :meth:`check_arbitrage`.

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

        theta_by_T: Dict[float, float] = {}
        theta_ref = None
        if isinstance(instance, (SSVI, ESSVI)):
            for T, g in groups:
                theta_by_T[T] = float(np.nanmin(g["iv"] ** 2 * g["maturity"]))
            theta_ref = float(np.median(list(theta_by_T.values())))

        slices = []
        for T, g in groups:
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
                kwargs["F"] = float(g["implied_forward"].iloc[0])
                kwargs.setdefault("beta", 0.5)
            params = calibrate_slice(g, instance, **kwargs)
            if params is None:
                logger.warning(
                    f"VolSurface.fit: slice T={T:g} failed to calibrate; skipping"
                )
                continue
            slices.append((T, params))
        if not slices:
            raise ValueError("VolSurface.fit: no slice calibrated successfully")
        return cls(instance, slices, r=r)

    # ── Slice access ─────────────────────────────────────────────────

    @property
    def maturities(self) -> np.ndarray:
        """Fitted maturities, ascending."""
        return np.array([T for T, _ in self._slices])

    def _slice(self, maturity) -> Tuple[float, Dict[str, float]]:
        T = float(maturity)
        for Ti, params in self._slices:
            if abs(Ti - T) <= 1e-12 * max(1.0, abs(T)):
                return Ti, params
        fitted = ", ".join(f"{Ti:g}" for Ti, _ in self._slices)
        raise ValueError(
            f"maturity {T:g} is not a fitted slice (fitted: {fitted}); "
            "maturity interpolation between slices arrives in a later release"
        )

    def params(self, maturity) -> Dict[str, float]:
        """Calibrated parameter dict of the slice at ``maturity`` (a copy)."""
        return dict(self._slice(maturity)[1])

    def forward(self, maturity) -> float:
        """Forward price of the slice at ``maturity``."""
        return float(self._slice(maturity)[1]["forward"])

    # ── Evaluation ───────────────────────────────────────────────────

    def total_variance(self, k, maturity) -> np.ndarray:
        """Total variance w(k) on the slice at ``maturity``."""
        _, params = self._slice(maturity)
        values = self.model.total_variance(np.atleast_1d(np.asarray(k, dtype=np.float64)), params)
        return _shape_like(values, k)

    def iv(self, strike, maturity):
        """Implied volatility at absolute strike(s) on the slice at ``maturity``."""
        T, params = self._slice(maturity)
        K = np.atleast_1d(np.asarray(strike, dtype=np.float64))
        if np.any(K <= 0):
            raise ValueError("strikes must be positive")
        k = np.log(K / params["forward"])
        w = self.model.total_variance(k, params)
        sigma = np.sqrt(np.maximum(w, 0.0) / T)
        return _shape_like(sigma, strike)

    def atm_vol(self, maturity) -> float:
        """At-the-money (k = 0) implied volatility of the slice."""
        T, params = self._slice(maturity)
        w0 = float(self.model.total_variance(np.array([0.0]), params)[0])
        return float(np.sqrt(max(w0, 0.0) / T))

    def skew(self, maturity) -> float:
        """ATM total-variance skew dw/dk at k = 0."""
        _, params = self._slice(maturity)
        return float(self.model.dw_dk(np.array([0.0]), params)[0])

    def curvature(self, maturity) -> float:
        """ATM total-variance curvature d2w/dk2 at k = 0."""
        _, params = self._slice(maturity)
        return float(self.model.d2w_dk2(np.array([0.0]), params)[0])

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
        T, params = self._slice(maturity)
        F = float(params["forward"])
        K = np.atleast_1d(np.asarray(strike, dtype=np.float64))
        if np.any(K <= 0):
            raise ValueError("strikes must be positive")
        k = np.log(K / F)
        w = np.maximum(self.model.total_variance(k, params), 1e-16)
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
