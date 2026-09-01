# src/pysvi/models.py
"""
Core parametrizations for stochastic volatility inspired IV surfaces.
NumPy implementations with optional numba acceleration (see use_numba).
Extensible via Parametrization ABC.
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Dict, Optional
from numpy.typing import NDArray
from loguru import logger

from . import _kernels


def numba_available() -> bool:
    """True if the optional numba dependency is installed.

    Install it with ``pip install "svi-py[numba]"``.
    """
    return _kernels.numba_available()


def use_numba(enabled: bool = True) -> None:
    """Toggle the numba-accelerated kernels at runtime.

    The numba backend is enabled automatically when numba is installed
    (disable at import time with the environment variable
    ``PYSVI_NUMBA=0``). All kernels have a pure-NumPy twin, so toggling
    changes speed, not results — outputs agree to within floating-point
    rounding (jitted kernels use fastmath).

    Parameters
    ----------
    enabled : bool, default True
        True activates jitted kernels; False falls back to pure NumPy.

    Raises
    ------
    ImportError
        If enabling is requested but numba is not installed.
    """
    _kernels.use_numba(enabled)


class ArbitrageFreedom(Flag):
    """Configurable arbitrage-freeness constraints for IV parametrizations.

    Combine flags with ``|`` to enforce multiple conditions simultaneously.

    Attributes
    ----------
    QUASI : default
        Soft parameter-bound constraints only (b > 0, abs(rho) < 1, sigma > 0).
    NO_BUTTERFLY : flag
        Enforce non-negative density g(k) >= 0 across strikes (no static arb).
    NO_CALENDAR : flag
        Enforce non-decreasing total variance in maturity (no calendar spread arb).
    """

    QUASI = 0
    NO_BUTTERFLY = auto()
    NO_CALENDAR = auto()


def _as_f64(k) -> NDArray[np.float64]:
    """Cast log-moneyness input to a float64 array for kernel dispatch."""
    return np.asarray(k, dtype=np.float64)


def svi_total_variance(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray:
    """Raw SVI total variance w(k)."""
    return _kernels.resolve("svi_w")(_as_f64(k), a, b, rho, m, sigma)


def ssvi_total_variance(
    k: np.ndarray, theta: float, rho: float, phi_theta: float
) -> np.ndarray:
    """SSVI total variance w(k)."""
    return _kernels.resolve("ssvi_w")(_as_f64(k), theta, rho, phi_theta)


def essvi_total_variance(
    k: np.ndarray, theta: float, rho_theta: float, phi_theta: float
) -> np.ndarray:
    """eSSVI total variance w(k)."""
    return _kernels.resolve("essvi_w")(_as_f64(k), theta, rho_theta, phi_theta)


def jw_total_variance(
    k: np.ndarray, v_t: float, psi_t: float, p_t: float, c_t: float, v_tilde_t: float, T: float
) -> np.ndarray:
    """Jump-wings SVI total variance w(k; T).

    The jump-wings parametrization [Gatheral 2004] converts to raw SVI via::

        b = (p_t + c_t) / 2
        rho = 1 - p_t / b   (equivalently (c_t - p_t) / (c_t + p_t))
        beta = rho - 2 * psi_t * sqrt(T) / b
        alpha = sign(beta) * sqrt(1 / (beta^2) - 1)   when abs(beta) < 1
        m = (v_t - v_tilde_t) * T / (b * (-rho + sign(alpha) * sqrt(1 + alpha^2) - alpha * sqrt(1 - rho^2)))
        sigma = alpha * m
        a = v_tilde_t * T - b * sigma * sqrt(1 - rho^2)

    Parameters
    ----------
    k : array
        Log-moneyness log(K/F).
    v_t : float
        ATM variance (annualised), v_t = sigma_ATM^2.
    psi_t : float
        ATM skew dw/dk|_{k=0} / (2 T).
    p_t : float
        Left-wing slope (put wing), p_t >= 0.
    c_t : float
        Right-wing slope (call wing), c_t >= 0.
    v_tilde_t : float
        Minimum implied variance, v_tilde_t > 0.
    T : float
        Time to expiry in years.

    Returns
    -------
    array
        Total variance w(k) = sigma^2(k) * T.
    """
    return _kernels.resolve("jw_w")(_as_f64(k), v_t, psi_t, p_t, c_t, v_tilde_t, T)


def natural_total_variance(
    k: np.ndarray, delta: float, mu: float, rho: float, omega: float, zeta: float
) -> np.ndarray:
    """Natural SVI total variance w(k) [Gatheral & Jacquier 2014].

    ::

        w(k) = Δ + ω/2 {1 + ζρ(k − μ) + sqrt[(ζ(k − μ) + ρ)² + (1 − ρ²)]}

    Bijective with raw SVI (see :func:`natural_to_raw` /
    :func:`raw_to_natural`); the natural parameters map more directly to
    ATM level, skew, and curvature.

    Parameters
    ----------
    k : array
        Log-moneyness log(K/F).
    delta : float
        Vertical variance shift (minimum-variance level).
    mu : float
        Log-moneyness translation of the smile.
    rho : float
        Skew (correlation), abs(rho) < 1.
    omega : float
        Overall variance scale, omega >= 0.
    zeta : float
        Curvature / smile-width scale, zeta > 0.

    Returns
    -------
    array
        Total variance w(k) = sigma^2(k) * T.
    """
    return _kernels.resolve("natural_w")(_as_f64(k), delta, mu, rho, omega, zeta)


def natural_to_raw(
    delta: float, mu: float, rho: float, omega: float, zeta: float
) -> Dict[str, float]:
    """Natural SVI parameters -> raw SVI parameters.

    ::

        a = Δ + ω(1 − ρ²)/2,   b = ωζ/2,   ρ = ρ,
        m = μ − ρ/ζ,           σ = sqrt(1 − ρ²)/ζ

    Requires zeta > 0 and abs(rho) < 1.

    Returns
    -------
    dict
        {'a', 'b', 'rho', 'm', 'sigma'}.
    """
    a, b, rho_r, m, sigma = _kernels.resolve("natural_convert")(
        delta, mu, rho, omega, zeta
    )
    return {
        "a": float(a), "b": float(b), "rho": float(rho_r),
        "m": float(m), "sigma": float(sigma),
    }


def raw_to_natural(
    a: float, b: float, rho: float, m: float, sigma: float
) -> Dict[str, float]:
    """Raw SVI parameters -> natural SVI parameters (inverse bijection).

    ::

        ζ = sqrt(1 − ρ²)/σ,   ω = 2bσ/sqrt(1 − ρ²),   ρ = ρ,
        μ = m + ρσ/sqrt(1 − ρ²),   Δ = a − bσ sqrt(1 − ρ²)

    Requires sigma > 0 and abs(rho) < 1.

    Returns
    -------
    dict
        {'delta', 'mu', 'rho', 'omega', 'zeta'}.
    """
    root = np.sqrt(1.0 - rho * rho)
    zeta = root / sigma
    omega = 2.0 * b * sigma / root
    mu = m + rho * sigma / root
    delta = a - b * sigma * root
    return {
        "delta": float(delta), "mu": float(mu), "rho": float(rho),
        "omega": float(omega), "zeta": float(zeta),
    }


def sabr_implied_vol(
    k: np.ndarray, alpha: float, beta: float, rho: float, nu: float,
    F: float, T: float,
) -> np.ndarray:
    """SABR lognormal (Black) implied volatility via Hagan et al. (2002).

    The SABR model [Hagan, Kumar, Lesniewski, Woodward 2002] assumes
    forward dynamics::

        dF = alpha * F^beta dW1,   d(alpha) = nu * alpha dW2,
        d<W1, W2> = rho dt

    and admits the asymptotic implied-vol expansion (HKLW formula)::

        sigma_B(K, F) = alpha / [(FK)^((1-beta)/2) * D(L)] * (z / x(z))
                        * {1 + [ (1-beta)^2 alpha^2 / (24 (FK)^(1-beta))
                               + rho beta nu alpha / (4 (FK)^((1-beta)/2))
                               + (2 - 3 rho^2) nu^2 / 24 ] T}

    where::

        L    = log(F/K)
        D(L) = 1 + (1-beta)^2/24 L^2 + (1-beta)^4/1920 L^4
        z    = (nu/alpha) (FK)^((1-beta)/2) L
        x(z) = log[(sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)]

    At the money z -> 0 and z/x(z) -> 1; the singularity is handled via
    the Taylor expansion z/x(z) = 1 - rho z / 2 + O(z^2).

    Rendered formulas: https://pysvi.readthedocs.io/en/latest/models/sabr.html

    Parameters
    ----------
    k : array
        Log-moneyness log(K/F).
    alpha : float
        Initial (ATM-like) volatility level, alpha > 0.
    beta : float
        CEV exponent in [0, 1]. Fixed by market convention, not fitted:
        beta = 1 (lognormal) for FX/equity, beta ~ 0.5 for interest rates,
        beta = 0 (normal) for spread-like underlyings.
    rho : float
        Spot/vol correlation, abs(rho) < 1.
    nu : float
        Vol-of-vol, nu >= 0.
    F : float
        Forward price of the underlying, F > 0.
    T : float
        Time to expiry in years, T > 0.

    Returns
    -------
    array
        Lognormal implied volatilities sigma_B(k).
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"SABR beta must be in [0, 1], got {beta}")
    if alpha <= 0:
        raise ValueError(f"SABR alpha must be positive, got {alpha}")
    if F <= 0:
        raise ValueError(f"SABR forward F must be positive, got {F}")

    return _kernels.resolve("sabr_vol")(_as_f64(k), alpha, beta, rho, nu, F, T)


def sabr_total_variance(
    k: np.ndarray, alpha: float, beta: float, rho: float, nu: float,
    F: float, T: float,
) -> np.ndarray:
    """SABR total variance w(k) = sigma_B(k)^2 * T via the Hagan expansion.

    See :func:`sabr_implied_vol` for the model and parameter definitions.
    """
    sigma = sabr_implied_vol(k, alpha, beta, rho, nu, F, T)
    return sigma**2 * T


def _finite_diff_derivatives(
    k_grid: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical w'(k), w''(k) on a uniform grid via central differences.

    Used for parametrizations without tractable analytic derivatives
    (e.g. SABR) when evaluating the butterfly density penalty.
    Central differences in the interior, one-sided at the edges.
    """
    return _kernels.resolve("finite_diff")(_as_f64(k_grid), _as_f64(w))


def _butterfly_penalty(
    k: np.ndarray, w: np.ndarray, dw: np.ndarray, d2w: np.ndarray
) -> float:
    """Penalty for butterfly arbitrage violations.

    The call price density is proportional to g(k) where

        g(k) = (1 - k w'/(2w))^2 - (w')^2/4 (1/w + 1/4) + w''/2

    Butterfly arbitrage is absent iff g(k) >= 0 for all k.
    Returns a penalty proportional to the integral of max(-g, 0).
    """
    return float(_kernels.resolve("butterfly")(k, w, dw, d2w))


def _svi_derivatives(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute w, w', w'' for raw SVI parametrization."""
    return _kernels.resolve("svi_derivs")(k, a, b, rho, m, sigma)


def _ssvi_derivatives(
    k: np.ndarray, theta: float, rho: float, phi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute w, w', w'' for SSVI/eSSVI parametrization."""
    return _kernels.resolve("ssvi_derivs")(k, theta, rho, phi)


def _calendar_penalty(
    k_grid: np.ndarray, w_current: np.ndarray, w_prev: np.ndarray
) -> float:
    """Penalty for calendar spread arbitrage violations.

    Calendar arbitrage is absent iff total variance is non-decreasing in
    maturity for every log-moneyness k.  That is, w(k, T2) >= w(k, T1)
    for T2 > T1.

    Parameters
    ----------
    k_grid : array
        Common evaluation grid.
    w_current : array
        Total variance of the current (later) slice on k_grid.
    w_prev : array
        Total variance of the prior (earlier) slice on k_grid.

    Returns sum-of-squares of violations.
    """
    return float(_kernels.resolve("calendar")(w_current, w_prev))


def _penalty_grid(k) -> NDArray[np.float64]:
    """The NO_BUTTERFLY / NO_CALENDAR penalty evaluation grid.

    The data range widened by 0.5 in log-moneyness, 200 points. A
    ``w_prev`` array passed to ``calibrate`` must be evaluated on this
    grid; ``calibrate_surface`` uses this helper for its chaining.
    """
    return np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 200)


def _prepare_objective_inputs(k, w_target, arbitrage_condition, kwargs):
    """Common calibration setup for the fused objective kernels.

    Casts inputs to float64, builds the penalty evaluation grid (empty when
    no grid-based constraint is active), and unpacks the arbitrage flags and
    optional prior-slice total variance into kernel-friendly values.
    """
    k = np.asarray(k, dtype=np.float64)
    w_target = np.asarray(w_target, dtype=np.float64)
    check_butterfly = ArbitrageFreedom.NO_BUTTERFLY in arbitrage_condition
    check_calendar = ArbitrageFreedom.NO_CALENDAR in arbitrage_condition
    if check_butterfly or check_calendar:
        k_grid = _penalty_grid(k)
    else:
        k_grid = np.empty(0)
    w_prev = kwargs.get("w_prev")
    has_prev = w_prev is not None
    w_prev_arr = np.asarray(w_prev, dtype=np.float64) if has_prev else np.empty(0)
    return k, w_target, k_grid, w_prev_arr, has_prev, check_butterfly, check_calendar


#: Residual spaces for calibration; codes shared with the loss kernels.
_OBJECTIVE_CODES = {
    "total_variance": 0,
    "implied_vol": 1,
    "price": 2,
    "vega_weighted": 3,
    "bid_ask": 4,
}

#: Robust losses (scipy.least_squares convention).
_LOSS_CODES = {"l2": 0, "huber": 1, "soft_l1": 2, "cauchy": 3}


def _prepare_loss_inputs(k, w_target, kwargs):
    """Resolve the calibration residual space and robust loss from kwargs.

    Returns (mode, loss_code, weights, w_lo, w_hi). Vega weights are
    precomputed here from the market data (Black vega up to per-slice
    constants, normalized to mean one); the bid/ask band arrives as
    total-variance arrays via the 'w_bid'/'w_ask' kwargs.
    """
    objective = kwargs.get("objective", "total_variance")
    loss = kwargs.get("loss", "l2")
    if objective not in _OBJECTIVE_CODES:
        raise ValueError(
            f"unknown objective {objective!r}; choose from {sorted(_OBJECTIVE_CODES)}"
        )
    if loss not in _LOSS_CODES:
        raise ValueError(
            f"unknown loss {loss!r}; choose from {sorted(_LOSS_CODES)}"
        )
    mode = _OBJECTIVE_CODES[objective]
    loss_code = _LOSS_CODES[loss]
    empty = np.empty(0)
    weights, w_lo, w_hi = empty, empty, empty
    if objective == "vega_weighted":
        s = np.sqrt(np.maximum(w_target, 1e-16))
        d1 = -k / s + 0.5 * s
        weights = np.exp(-0.5 * d1 * d1)
        mean_w = float(np.mean(weights))
        weights = weights / mean_w if mean_w > 0 else np.ones_like(w_target)
    elif objective == "bid_ask":
        if "w_bid" not in kwargs or "w_ask" not in kwargs:
            raise ValueError(
                "objective='bid_ask' requires 'w_bid' and 'w_ask' kwargs: "
                "total variance of the bid and ask quotes (iv_bid^2 T, iv_ask^2 T)"
            )
        w_lo = np.asarray(kwargs["w_bid"], dtype=np.float64)
        w_hi = np.asarray(kwargs["w_ask"], dtype=np.float64)
        if w_lo.shape != w_target.shape or w_hi.shape != w_target.shape:
            raise ValueError("w_bid/w_ask must have the same shape as the quotes")
    return mode, loss_code, weights, w_lo, w_hi


def _mad_scale(k, w_model, w_target, mode, weights, w_lo, w_hi):
    """1.4826 * MAD of the mode-space residuals at w_model.

    Floored at 1e-6 of the data scale: on (near-)clean data the MAD
    collapses to rounding noise, and an absolute-tiny f_scale would put
    every residual on the loss plateau, degrading the optimizer. At the
    relative floor the robust losses stay in their quadratic region and
    behave like l2, which is the correct clean-data limit.
    """
    r = _kernels.resolve("residuals")(k, w_model, w_target, mode, weights, w_lo, w_hi)
    mad = float(np.median(np.abs(r - np.median(r))))
    floor = 1e-6 * float(np.median(np.abs(w_target))) if w_target.size else 0.0
    return max(1.4826 * mad, floor, 1e-12)


def _resolve_f_scale(kwargs, k, w0, w_target, mode, loss_code, weights, w_lo, w_hi,
                     pilot=None):
    """Robust-loss scale: explicit kwarg, else a data-driven default.

    l2 needs no scale (1.0). For bid_ask the natural scale is the band
    width. Otherwise the default is 1.4826 * MAD of the mode-space
    residuals at a pilot l2 fit (so genuine outliers stand out against
    the fitted noise level, not against initial-guess error); when the
    pilot fails, the residuals at the initial guess w0 are used instead.
    """
    f_scale = kwargs.get("f_scale")
    if f_scale is not None:
        return float(f_scale)
    if loss_code == _LOSS_CODES["l2"]:
        return 1.0
    if mode == _OBJECTIVE_CODES["bid_ask"]:
        return max(float(np.median(w_hi - w_lo)), 1e-12)
    if pilot is not None:
        scale = pilot()
        if scale is not None:
            return scale
    return _mad_scale(k, w0, w_target, mode, weights, w_lo, w_hi)


def _initialization(kwargs, supports_jump_wings: bool = False) -> str:
    """Validate the 'initialization' kwarg."""
    init = kwargs.get("initialization", "default")
    if init not in ("default", "jump_wings", "multi_start"):
        raise ValueError(
            f"unknown initialization {init!r}; choose 'default', 'jump_wings', "
            "or 'multi_start'"
        )
    if init == "jump_wings" and not supports_jump_wings:
        raise ValueError(
            "initialization='jump_wings' is only available for SVI and NaturalSVI"
        )
    return init


def _wing_readoff_x0(k, w_target):
    """Data-driven raw-SVI start [a, b, rho, m, sigma] from ATM and wings.

    Jump-wings-style readoff: wing slopes from least-squares fits to the
    outer 20% of points on each side, skew from their asymmetry, vertex
    from the minimum-variance strike.
    """
    order = np.argsort(k)
    ks, ws = k[order], w_target[order]
    n_wing = max(2, ks.size // 5)

    def _slope(x, y):
        xc = x - x.mean()
        denom = float(np.sum(xc * xc))
        return float(np.sum(xc * (y - y.mean())) / denom) if denom > 0 else 0.0

    p_hat = max(-_slope(ks[:n_wing], ws[:n_wing]), 1e-4)   # put wing: w falls in k
    c_hat = max(_slope(ks[-n_wing:], ws[-n_wing:]), 1e-4)  # call wing: w rises in k
    b0 = 0.5 * (p_hat + c_hat)
    rho0 = float(np.clip((c_hat - p_hat) / (c_hat + p_hat), -0.9, 0.9))
    m0 = float(ks[int(np.argmin(ws))])
    sigma0 = max(float(np.std(ks)) / 2.0, 0.05)
    a0 = float(np.nanmin(ws)) - b0 * sigma0 * np.sqrt(1.0 - rho0 * rho0)
    return np.array([a0, b0, rho0, m0, sigma0])


def _multistart_variants(
    x0, rho_idx, scale_idx,
    rho_values=(-0.7, -0.3, 0.0, 0.3, 0.7),
    scale_values=(0.5, 1.0, 2.0),
):
    """Deterministic start grid: the default start plus variations of the
    skew-like coordinate and a width-like coordinate scaling."""
    base = np.asarray(x0, dtype=np.float64)
    starts = [base]
    for rho in rho_values:
        for sc in scale_values:
            v = base.copy()
            v[rho_idx] = rho
            v[scale_idx] = base[scale_idx] * sc
            starts.append(v)
    return starts


def _tight_if_controls(init, mode, loss_code):
    """Tight L-BFGS-B options when any calibration control is active.

    The legacy default path (total_variance / l2 / default start) keeps
    scipy's default tolerances for backward-compatible fits; the new
    residual spaces and robust losses produce objective values orders of
    magnitude below scipy's relative ftol, and multi_start wants deep
    convergence before comparing basins.
    """
    if init != "default" or mode != 0 or loss_code != 0:
        return {"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000}
    return None


def _minimize_with_starts(objective, starts, bounds, lbfgs_options=None, nm_options=None):
    """L-BFGS-B from each start, keeping the best converged result.

    Falls back to Nelder-Mead from the first start when no start
    converges. Returns the scipy result, or None on total failure.
    """
    from scipy.optimize import minimize

    best = None
    for x0 in starts:
        res = minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options=lbfgs_options or {},
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is not None:
        return best
    res = minimize(
        objective, starts[0], method="Nelder-Mead",
        options=nm_options if nm_options is not None else {},
    )
    return res if res.success else None


class Parametrization(ABC):
    """Base class for IV surface parametrizations."""

    def __init__(
        self, arbitrage_condition: ArbitrageFreedom = ArbitrageFreedom.QUASI
    ) -> None:
        self.arbitrage_condition = arbitrage_condition

    @abstractmethod
    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """
        Calibrate parameters from log-moneyness k and total variance w_target.

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness values log(K/F).
        w_target : np.ndarray
            Observed total variance values sigma_mkt^2 * T.
        **kwargs :
            Extra model-specific arguments (e.g. theta for SSVI/eSSVI),
            plus the common calibration controls accepted by every
            iterative model:

            * objective : str, default 'total_variance' — residual space:
              'total_variance', 'implied_vol', 'price' (Black call),
              'vega_weighted', or 'bid_ask' (requires 'w_bid'/'w_ask'
              arrays, the total variance of the bid and ask quotes).
            * loss : str, default 'l2' — 'l2', 'huber', 'soft_l1', or
              'cauchy' (scipy.least_squares convention).
            * f_scale : float, optional — robust-loss scale; defaults to
              1.4826 * MAD of the residuals at a pilot l2 fit.
            * initialization : str, default 'default' — 'default',
              'jump_wings' (SVI/NaturalSVI only: data-driven wing
              readoff), or 'multi_start' (deterministic start grid,
              best converged result wins).

        Returns
        -------
        dict or None
            Mapping of parameter names to floats, or None on failure.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.calibrate() must be implemented by subclasses."
        )

    @abstractmethod
    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """
        Compute model total variance w(k) given parameters.

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness values log(K/F).
        params : dict
            Calibrated parameter dictionary for this parametrization.

        Returns
        -------
        np.ndarray
            Total variance values w(k).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.total_variance() must be implemented by subclasses."
        )

    def _pilot_f_scale(self, k, w_target, kwargs, mode, weights, w_lo, w_hi):
        """Robust-loss scale from a pilot l2 fit in the same residual space."""
        pilot_kwargs = {
            key: val for key, val in kwargs.items()
            if key not in ("loss", "f_scale", "initialization")
        }
        pilot = self.calibrate(k, w_target, **pilot_kwargs)
        if pilot is None:
            return None
        w_fit = self.total_variance(k, pilot)
        return _mad_scale(k, w_fit, w_target, mode, weights, w_lo, w_hi)

    #: Step for the default finite-difference derivatives (central,
    #: second-order: truncation O(h^2), roundoff on w'' ~ eps/h^2). Set it
    #: on an instance to trade truncation against roundoff; note the density
    #: noise this induces for finite-difference models (SABR, DirectSVI) can
    #: reach ~1e-2, far above the default diagnostics tolerance.
    fd_step: float = 1e-5

    def derivatives(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Total variance and its first two strike derivatives, (w, w', w'').

        The base implementation uses central finite differences with step
        :attr:`fd_step` on :meth:`total_variance`; parametrizations with
        tractable analytic derivatives override it (SVI, SSVI, eSSVI,
        jump-wings). SABR and DirectSVI use this finite-difference default.

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness values log(K/F).
        params : dict
            Calibrated parameter dictionary for this parametrization.

        Returns
        -------
        tuple of np.ndarray
            (w(k), w'(k), w''(k)).
        """
        k = np.asarray(k, dtype=np.float64)
        h = self.fd_step
        w = self.total_variance(k, params)
        w_up = self.total_variance(k + h, params)
        w_dn = self.total_variance(k - h, params)
        dw = (w_up - w_dn) / (2.0 * h)
        d2w = (w_up - 2.0 * w + w_dn) / (h * h)
        return w, dw, d2w

    def dw_dk(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """First derivative of total variance, w'(k). See :meth:`derivatives`."""
        return self.derivatives(k, params)[1]

    def d2w_dk2(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """Second derivative of total variance, w''(k). See :meth:`derivatives`."""
        return self.derivatives(k, params)[2]

    def wing_slopes(
        self, params: Dict[str, float]
    ) -> Optional[tuple[float, float]]:
        """Asymptotic total-variance wing slopes (left, right).

        Returns lim dw/d(abs(k)) on each wing where a closed form exists
        (the SVI family), else None — callers should fall back to
        measuring dw/dk at finite k, which underestimates the asymptote
        for convex w. Used by the Lee-bound check in the diagnostics.
        """
        return None

    def density(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """Risk-neutral density factor g(k).

        ::

            g(k) = (1 - k w'/(2w))^2 - (w')^2/4 (1/w + 1/4) + w''/2

        The slice is free of butterfly arbitrage iff g(k) >= 0 for all k
        [Gatheral & Jacquier 2014]. Uses :meth:`derivatives`, so models
        with analytic derivatives get an analytic density. Only valid
        where w(k) > 0; non-positive total variance produces NaN/inf or
        meaningless values rather than raising — validate w separately
        (the diagnostics module does).

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness values log(K/F).
        params : dict
            Calibrated parameter dictionary for this parametrization.

        Returns
        -------
        np.ndarray
            g(k) at each input point.
        """
        k = np.asarray(k, dtype=np.float64)
        w, dw, d2w = self.derivatives(k, params)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _kernels.resolve("density_g")(k, w, dw, d2w)


class SVI(Parametrization):
    """Raw SVI total variance parametrization [Gatheral 2004].

    ::

        w(k) = a + b {ρ(k-m) + sqrt[(k-m)² + σ²]}

    No-arbitrage constraints softly enforced via bounds/penalties:

    * b > 0 (positive slope)
    * abs(ρ) < 1 (correlation)
    * σ > 0 (vol of vol)

    Calibrates via L-BFGS-B (bounded) → Nelder-Mead fallback.
    Initial guess: ATM a, median m, wing-informed b/σ.

    Rendered formulas: https://pysvi.readthedocs.io/en/latest/models/svi.html
    """

    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """Minimize MSE(w_model(k), w_target) subject to constraints.

        Parameters
        ----------
        k : NDArray[np.float64]
            Log-moneyness array.
        w_target : NDArray[np.float64]
            Market total variances σ_mkt²T.

        Returns
        -------
        Dict[str, float] or None
            {'a', 'b', 'rho', 'm', 'sigma'} or None (opt failed).
        """

        k, w_target, k_grid, w_prev_arr, has_prev, check_butterfly, check_calendar = (
            _prepare_objective_inputs(k, w_target, self.arbitrage_condition, kwargs)
        )
        mode, loss_code, weights, w_lo, w_hi = _prepare_loss_inputs(k, w_target, kwargs)
        core = _kernels.resolve("svi_obj")

        def objective(params):
            return core(
                np.asarray(params, dtype=np.float64), k, w_target,
                k_grid, w_prev_arr, check_butterfly, check_calendar, has_prev,
                mode, weights, w_lo, w_hi, loss_code, f_scale,
            )

        init = _initialization(kwargs, supports_jump_wings=True)
        if init == "jump_wings":
            x0 = _wing_readoff_x0(k, w_target)
        else:
            a0 = float(np.nanmin(w_target))
            spread = float(np.nanmax(w_target) - a0)
            k_abs_max = float(np.max(np.abs(k)))
            denom = max(k_abs_max, 1.0)
            b0 = max(spread / denom, 1e-4)
            x0 = np.array([a0, b0, 0.0, float(np.median(k)), max(float(np.std(k)), 0.1)])

        bounds = [
            (None, None),
            (1e-8, None),
            (-0.999, 0.999),
            (None, None),
            (1e-8, None),
        ]

        f_scale = _resolve_f_scale(
            kwargs, k, svi_total_variance(k, *x0), w_target,
            mode, loss_code, weights, w_lo, w_hi,
            pilot=lambda: self._pilot_f_scale(
                k, w_target, kwargs, mode, weights, w_lo, w_hi
            ),
        )
        starts = _multistart_variants(x0, 2, 4) if init == "multi_start" else [x0]
        res = _minimize_with_starts(
            objective, starts, bounds,
            lbfgs_options=_tight_if_controls(init, mode, loss_code),
            nm_options={"maxiter": 2000},
        )
        if res is None:
            return None

        a, b, rho, m, sigma = res.x
        if b <= 0 or sigma <= 0 or abs(rho) >= 0.999:
            return None

        return {
            "a": float(a),
            "b": float(b),
            "rho": float(rho),
            "m": float(m),
            "sigma": float(sigma),
        }

    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """Evaluate w(k) = a + b{ρ(k-m) + sqrt[(k-m)² + σ²]}."""
        svi_params = {p: params[p] for p in ["a", "b", "rho", "m", "sigma"]}
        return svi_total_variance(k, **svi_params)

    def derivatives(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Analytic (w, w', w'') for raw SVI."""
        return _svi_derivatives(
            _as_f64(k), params["a"], params["b"], params["rho"],
            params["m"], params["sigma"],
        )

    def wing_slopes(
        self, params: Dict[str, float]
    ) -> Optional[tuple[float, float]]:
        """Asymptotic wing slopes: left b(1 - ρ), right b(1 + ρ)."""
        b, rho = params["b"], params["rho"]
        return b * (1.0 - rho), b * (1.0 + rho)


class NaturalSVI(Parametrization):
    """Natural SVI parametrization [Gatheral & Jacquier 2014].

    ::

        w(k) = Δ + ω/2 {1 + ζρ(k − μ) + sqrt[(ζ(k − μ) + ρ)² + (1 − ρ²)]}

        Δ : vertical variance shift        (unconstrained)
        μ : log-moneyness translation      (unconstrained)
        ρ : skew (correlation)             abs(ρ) < 1
        ω : overall variance scale         ω > 0
        ζ : curvature / smile-width scale  ζ > 0

    Same 5 degrees of freedom as raw SVI, connected by an explicit
    bijection (:func:`natural_to_raw` / :func:`raw_to_natural`), but the
    parameters map more directly to ATM level, skew, and curvature —
    often better behaved in calibration and useful as an initialisation
    coordinate system for raw SVI.

    Calibrates via L-BFGS-B (bounded) → Nelder-Mead fallback; evaluation
    and derivatives go through the raw-SVI equivalents.

    Rendered formulas: https://pysvi.readthedocs.io/en/latest/models/natural.html
    """

    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """Minimize MSE(w_model(k), w_target) subject to constraints.

        Parameters
        ----------
        k : NDArray[np.float64]
            Log-moneyness array.
        w_target : NDArray[np.float64]
            Market total variances σ_mkt²T.
        **kwargs
            Optional 'w_prev': prior slice total variance for calendar arb.

        Returns
        -------
        Dict[str, float] or None
            {'delta', 'mu', 'rho', 'omega', 'zeta'} or None (opt failed).
        """

        k, w_target, k_grid, w_prev_arr, has_prev, check_butterfly, check_calendar = (
            _prepare_objective_inputs(k, w_target, self.arbitrage_condition, kwargs)
        )
        mode, loss_code, weights, w_lo, w_hi = _prepare_loss_inputs(k, w_target, kwargs)
        core = _kernels.resolve("natural_obj")

        def objective(params):
            return core(
                np.asarray(params, dtype=np.float64), k, w_target,
                k_grid, w_prev_arr, check_butterfly, check_calendar, has_prev,
                mode, weights, w_lo, w_hi, loss_code, f_scale,
            )

        init = _initialization(kwargs, supports_jump_wings=True)
        if init == "jump_wings":
            raw0 = _wing_readoff_x0(k, w_target)
            nat0 = raw_to_natural(*raw0)
            x0 = np.array([nat0["delta"], nat0["mu"], nat0["rho"],
                           nat0["omega"], nat0["zeta"]])
        else:
            # Raw-SVI heuristics mapped through the bijection at rho = 0
            # (zeta = 1/sigma, omega = 2 b sigma, mu = m, delta = a - b sigma)
            a0 = float(np.nanmin(w_target))
            spread = float(np.nanmax(w_target) - a0)
            k_abs_max = float(np.max(np.abs(k)))
            b0 = max(spread / max(k_abs_max, 1.0), 1e-4)
            sigma0 = max(float(np.std(k)), 0.1)
            x0 = np.array([
                a0 - b0 * sigma0,             # delta
                float(np.median(k)),          # mu
                0.0,                          # rho
                max(2.0 * b0 * sigma0, 1e-4), # omega
                1.0 / sigma0,                 # zeta
            ])

        bounds = [
            (None, None),      # delta
            (None, None),      # mu
            (-0.999, 0.999),   # abs(rho) < 1
            (1e-8, None),      # omega > 0
            (1e-8, None),      # zeta > 0
        ]

        f_scale = _resolve_f_scale(
            kwargs, k, natural_total_variance(k, *x0), w_target,
            mode, loss_code, weights, w_lo, w_hi,
            pilot=lambda: self._pilot_f_scale(
                k, w_target, kwargs, mode, weights, w_lo, w_hi
            ),
        )
        starts = _multistart_variants(x0, 2, 4) if init == "multi_start" else [x0]
        # Tight ftol/gtol as for SABR: total-variance MSEs are O(1e-8) even
        # mid-fit, so scipy's default relative ftol stops too early.
        res = _minimize_with_starts(
            objective, starts, bounds,
            lbfgs_options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000},
            nm_options={"maxiter": 2000, "fatol": 1e-14, "xatol": 1e-10},
        )
        if res is None:
            return None

        delta, mu, rho, omega, zeta = res.x
        if omega <= 0 or zeta <= 0 or abs(rho) >= 0.999:
            return None

        return {
            "delta": float(delta),
            "mu": float(mu),
            "rho": float(rho),
            "omega": float(omega),
            "zeta": float(zeta),
        }

    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """Evaluate natural SVI w(k) via the raw-SVI equivalents."""
        return natural_total_variance(
            k, params["delta"], params["mu"], params["rho"],
            params["omega"], params["zeta"],
        )

    @staticmethod
    def _raw_svi(params: Dict[str, float]) -> tuple[float, float, float, float, float]:
        return _kernels.resolve("natural_convert")(
            params["delta"], params["mu"], params["rho"],
            params["omega"], params["zeta"],
        )

    def derivatives(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Analytic (w, w', w'') via the raw-SVI equivalent parameters."""
        return _svi_derivatives(_as_f64(k), *self._raw_svi(params))

    def wing_slopes(
        self, params: Dict[str, float]
    ) -> Optional[tuple[float, float]]:
        """Asymptotic wing slopes via the raw-SVI equivalents: b(1 ∓ ρ)."""
        _, b, rho, _, _ = self._raw_svi(params)
        return b * (1.0 - rho), b * (1.0 + rho)


class SSVI(Parametrization):
    """Surface-consistent SSVI [Gatheral & Jacquier 2014].

    ::

        w(k;θ) = θ/2 [1 + ρ φ(θ) k + sqrt{(φ(θ) k + ρ)² + (1-ρ²)}]

        θ    = ATM total variance (fixed per slice, typically σ_ATM² T)
        φ(θ) = η / sqrt(θ) - curvature scale independent of ATM level

    Guarantees no butterfly arbitrage across strikes for fixed θ.
    Calibrates only ρ, η (2 params) given θ.

    Rendered formulas: https://pysvi.readthedocs.io/en/latest/models/ssvi.html
    """

    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """Fit ρ, η minimizing MSE(w_model, w_target) for fixed θ.

        Parameters
        ----------
        k : NDArray[np.float64]
            Log-moneyness.
        w_target : NDArray[np.float64]
            Market total variances.
        **kwargs
            Must contain 'theta': ATM w_ATM.

        Returns
        -------
        Dict[str, float] or None
            {'rho', 'eta', 'theta'} or None.
        """
        theta = float(kwargs["theta"])

        k, w_target, k_grid, w_prev_arr, has_prev, check_butterfly, check_calendar = (
            _prepare_objective_inputs(k, w_target, self.arbitrage_condition, kwargs)
        )
        mode, loss_code, weights, w_lo, w_hi = _prepare_loss_inputs(k, w_target, kwargs)
        core = _kernels.resolve("ssvi_obj")

        def objective(params):
            return core(
                np.asarray(params, dtype=np.float64), k, w_target, theta,
                k_grid, w_prev_arr, check_butterfly, check_calendar, has_prev,
                mode, weights, w_lo, w_hi, loss_code, f_scale,
            )

        init = _initialization(kwargs)
        x0 = np.array([0.0, 1.0])
        bounds = [(-0.999, 0.999), (1e-8, None)]

        f_scale = _resolve_f_scale(
            kwargs, k,
            ssvi_total_variance(k, theta, x0[0], x0[1] / np.sqrt(theta)),
            w_target, mode, loss_code, weights, w_lo, w_hi,
            pilot=lambda: self._pilot_f_scale(
                k, w_target, kwargs, mode, weights, w_lo, w_hi
            ),
        )
        starts = _multistart_variants(x0, 0, 1) if init == "multi_start" else [x0]
        res = _minimize_with_starts(
            objective, starts, bounds,
            lbfgs_options=_tight_if_controls(init, mode, loss_code),
        )
        if res is None:
            return None

        rho, eta = res.x
        if eta <= 0 or abs(rho) >= 0.999:
            return None

        return {"rho": float(rho), "eta": float(eta), "theta": float(theta)}

    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        theta = params["theta"]
        phi_theta = params["eta"] / np.sqrt(theta)
        return ssvi_total_variance(k, theta, params["rho"], phi_theta)

    def derivatives(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Analytic (w, w', w'') for SSVI."""
        theta = params["theta"]
        return _ssvi_derivatives(
            _as_f64(k), theta, params["rho"], params["eta"] / np.sqrt(theta)
        )

    def wing_slopes(
        self, params: Dict[str, float]
    ) -> Optional[tuple[float, float]]:
        """Asymptotic wing slopes: θφ(1 ∓ ρ)/2."""
        theta, rho = params["theta"], params["rho"]
        phi = params["eta"] / np.sqrt(theta)
        return 0.5 * theta * phi * (1.0 - rho), 0.5 * theta * phi * (1.0 + rho)


class ESSVI(Parametrization):
    """Extended SSVI with ρ(θ) parametrization.

    ::

        w(k;θ) = θ/2 [1 + ρ(θ) φ(θ) k + sqrt{(φ(θ) k + ρ(θ))² + (1-ρ(θ)²)}]

        ρ(θ) = clip(ρ₀ + ρ₁ (θ/θ_ref)^α, -0.999, 0.999)  ← term structure skew
        φ(θ) = η / sqrt(θ)                                ← curvature

    θ_ref smooths ρ across maturities (often median ATM θ). 4 params total.
    Enables realistic calendar skew evolution.

    Rendered formulas: https://pysvi.readthedocs.io/en/latest/models/essvi.html
    """

    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """Fit ρ₀, ρ₁, α, η given θ, θ_ref via penalized MSE.

        Heavy penalty on η≤0, mild on abs(ρ(θ))>0.95 for stability.

        Parameters
        ----------
        k : NDArray[np.float64]
            Log-moneyness.
        w_target : NDArray[np.float64]
            Total variances.
        **kwargs
            'theta': slice ATM w
            'theta_ref': reference θ (defaults to theta)

        Returns
        -------
        Dict[str, float] or None
            All params + computed 'rho_theta'.
        """
        theta = float(kwargs["theta"])
        theta_ref = kwargs["theta_ref"]


        if theta_ref is None:
            theta_ref = theta
        theta_ref = float(theta_ref)

        k, w_target, k_grid, w_prev_arr, has_prev, check_butterfly, check_calendar = (
            _prepare_objective_inputs(k, w_target, self.arbitrage_condition, kwargs)
        )
        mode, loss_code, weights, w_lo, w_hi = _prepare_loss_inputs(k, w_target, kwargs)
        core = _kernels.resolve("essvi_obj")

        def objective(params):
            return core(
                np.asarray(params, dtype=np.float64), k, w_target, theta, theta_ref,
                k_grid, w_prev_arr, check_butterfly, check_calendar, has_prev,
                mode, weights, w_lo, w_hi, loss_code, f_scale,
            )

        init = _initialization(kwargs)
        x0 = np.array([0.0, -0.5, 0.5, 1.0])
        bounds = [(-0.999, 0.999), (-2.0, 2.0), (-2.0, 2.0), (1e-8, None)]

        rho_init = self._rho_of(theta, theta_ref, x0[0], x0[1], x0[2])
        f_scale = _resolve_f_scale(
            kwargs, k,
            essvi_total_variance(k, theta, rho_init, x0[3] / np.sqrt(theta)),
            w_target, mode, loss_code, weights, w_lo, w_hi,
            pilot=lambda: self._pilot_f_scale(
                k, w_target, kwargs, mode, weights, w_lo, w_hi
            ),
        )
        starts = _multistart_variants(x0, 0, 3) if init == "multi_start" else [x0]
        res = _minimize_with_starts(
            objective, starts, bounds,
            lbfgs_options=_tight_if_controls(init, mode, loss_code),
        )
        if res is None:
            return None

        rho0, rho1, alpha, eta = res.x
        if eta <= 0:
            return None

        rho_theta = self._rho_of(theta, theta_ref, rho0, rho1, alpha)

        return {
            "rho0": float(rho0),
            "rho1": float(rho1),
            "alpha": float(alpha),
            "eta": float(eta),
            "theta": float(theta),
            "theta_ref": float(theta_ref),
            "rho_theta": float(rho_theta),
        }

    @staticmethod
    def _rho_of(
        theta: float, theta_ref: float, rho0: float, rho1: float, alpha: float
    ) -> float:
        """The ρ(θ) term structure: clip(ρ₀ + ρ₁(θ/θ_ref)^α, ±0.999)."""
        ratio = theta / max(theta_ref, 1e-12)
        return float(np.clip(rho0 + rho1 * ratio**alpha, -0.999, 0.999))

    @staticmethod
    def _rho_phi(params: Dict[str, float]) -> tuple[float, float, float]:
        """Resolve (theta, rho(theta), phi(theta)) from a params dict.

        A stored ``rho_theta`` takes precedence and suffices on its own;
        otherwise it is computed from (rho0, rho1, alpha, theta_ref).
        """
        theta = params["theta"]
        rho_theta = params.get("rho_theta")
        if rho_theta is None:
            rho_theta = ESSVI._rho_of(
                theta, params["theta_ref"],
                params["rho0"], params["rho1"], params["alpha"],
            )
        else:
            rho_theta = float(np.clip(rho_theta, -0.999, 0.999))
        phi_theta = params["eta"] / np.sqrt(theta)
        return theta, rho_theta, phi_theta

    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        theta, rho_theta, phi_theta = self._rho_phi(params)
        return essvi_total_variance(k, theta, rho_theta, phi_theta)

    def derivatives(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Analytic (w, w', w'') for eSSVI (SSVI form with ρ(θ))."""
        theta, rho_theta, phi_theta = self._rho_phi(params)
        return _ssvi_derivatives(_as_f64(k), theta, rho_theta, phi_theta)

    def wing_slopes(
        self, params: Dict[str, float]
    ) -> Optional[tuple[float, float]]:
        """Asymptotic wing slopes: θφ(1 ∓ ρ(θ))/2."""
        theta, rho_theta, phi_theta = self._rho_phi(params)
        return (
            0.5 * theta * phi_theta * (1.0 - rho_theta),
            0.5 * theta * phi_theta * (1.0 + rho_theta),
        )


class JumpWings(Parametrization):
    """SVI jump-wings parametrization [Gatheral 2004].

    Parameters are (v_t, psi_t, p_t, c_t, v_tilde_t) per slice at maturity T::

      v_t       : ATM variance sigma_ATM^2
      psi_t     : ATM skew (dw/dk at k=0) / (2T)
      p_t       : left (put) wing slope,  p_t >= 0
      c_t       : right (call) wing slope, c_t >= 0
      v_tilde_t : minimum implied variance, v_tilde_t > 0

    Internally converts to raw SVI (a, b, rho, m, sigma) for evaluation.
    Calibrates 5 jump-wings params via L-BFGS-B with Nelder-Mead fallback.

    Rendered formulas: https://pysvi.readthedocs.io/en/latest/models/jumpwings.html
    """

    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """Fit jump-wings params minimizing MSE(w_model, w_target).

        Parameters
        ----------
        k : NDArray[np.float64]
            Log-moneyness array.
        w_target : NDArray[np.float64]
            Market total variances.
        **kwargs
            Must contain 'T': time to expiry in years.
            Optional 'w_prev': prior slice total variance for calendar arb.

        Returns
        -------
        Dict[str, float] or None
            {'v_t', 'psi_t', 'p_t', 'c_t', 'v_tilde_t', 'T'} or None.
        """
        T = float(kwargs["T"])

        k, w_target, k_grid, w_prev_arr, has_prev, check_butterfly, check_calendar = (
            _prepare_objective_inputs(k, w_target, self.arbitrage_condition, kwargs)
        )
        mode, loss_code, weights, w_lo, w_hi = _prepare_loss_inputs(k, w_target, kwargs)
        core = _kernels.resolve("jw_obj")

        def objective(params):
            return core(
                np.asarray(params, dtype=np.float64), k, w_target, T,
                k_grid, w_prev_arr, check_butterfly, check_calendar, has_prev,
                mode, weights, w_lo, w_hi, loss_code, f_scale,
            )

        init = _initialization(kwargs)
        # Initial guess from market data
        v_t0 = float(np.interp(0.0, k, w_target)) / T if T > 0 else 0.04
        v_tilde_t0 = float(np.nanmin(w_target)) / T if T > 0 else 0.03
        x0 = np.array([max(v_t0, 1e-4), -0.1, 0.1, 0.1, max(v_tilde_t0, 1e-4)])

        bounds = [
            (1e-8, None),     # v_t > 0
            (-5.0, 5.0),      # psi_t
            (0.0, None),      # p_t >= 0
            (0.0, None),      # c_t >= 0
            (1e-8, None),     # v_tilde_t > 0
        ]

        f_scale = _resolve_f_scale(
            kwargs, k,
            jw_total_variance(k, x0[0], x0[1], x0[2], x0[3], x0[4], T),
            w_target, mode, loss_code, weights, w_lo, w_hi,
            pilot=lambda: self._pilot_f_scale(
                k, w_target, kwargs, mode, weights, w_lo, w_hi
            ),
        )
        starts = (
            _multistart_variants(x0, 1, 2, rho_values=(-0.5, -0.2, 0.0, 0.2, 0.5))
            if init == "multi_start" else [x0]
        )
        res = _minimize_with_starts(
            objective, starts, bounds,
            lbfgs_options=_tight_if_controls(init, mode, loss_code),
            nm_options={"maxiter": 2000},
        )
        if res is None:
            return None

        v_t, psi_t, p_t, c_t, v_tilde_t = res.x
        if v_t <= 0 or v_tilde_t <= 0 or p_t < 0 or c_t < 0:
            return None

        return {
            "v_t": float(v_t),
            "psi_t": float(psi_t),
            "p_t": float(p_t),
            "c_t": float(c_t),
            "v_tilde_t": float(v_tilde_t),
            "T": float(T),
        }

    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """Evaluate w(k) via jump-wings → raw SVI conversion."""
        return jw_total_variance(
            k, params["v_t"], params["psi_t"], params["p_t"],
            params["c_t"], params["v_tilde_t"], params["T"],
        )

    @staticmethod
    def _raw_svi(params: Dict[str, float]) -> Optional[tuple[float, float, float, float, float]]:
        """Jump-wings → raw SVI (a, b, rho, m, sigma); None for a flat slice.

        Delegates to the shared jw_convert kernel — the single source used
        by evaluation and the calibration objectives.
        """
        ok, a, b, rho, m, sigma = _kernels.resolve("jw_convert")(
            params["v_t"], params["psi_t"], params["p_t"],
            params["c_t"], params["v_tilde_t"], params["T"],
        )
        if not ok:
            return None  # degenerate: w(k) = v_t * T for all k
        return a, b, rho, m, sigma

    def derivatives(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Analytic (w, w', w'') via the raw-SVI equivalent parameters."""
        k = _as_f64(k)
        raw = self._raw_svi(params)
        if raw is None:
            w = np.full_like(k, params["v_t"] * params["T"])
            zero = np.zeros_like(k)
            return w, zero, zero.copy()
        return _svi_derivatives(k, *raw)

    def wing_slopes(
        self, params: Dict[str, float]
    ) -> Optional[tuple[float, float]]:
        """Asymptotic wing slopes via the raw-SVI equivalents: b(1 ∓ ρ)."""
        raw = self._raw_svi(params)
        if raw is None:
            return 0.0, 0.0
        _, b, rho, _, _ = raw
        return b * (1.0 - rho), b * (1.0 + rho)


def directsvi_fit(
    k: NDArray[np.float64], w: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Direct algebraic SVI fit via conic section eigenvalue problem.

    Linearises the SVI equation as a hyperbola in (x, y) = (k, w) space::

        z₀x² + z₁y² + z₂xy + z₃x + z₄y + z₅ = 0

    and solves for the 6 conic coefficients via a constrained eigenvalue
    problem (hyperbola constraint: z₂² − 4z₀z₁ > 0).

    Parameters
    ----------
    k : array
        Log-moneyness values.
    w : array
        Total variance values (market observed).

    Returns
    -------
    ndarray, shape (6,)
        Conic coefficients [z0, z1, z2, z3, z4, z5] normalised so z1 = 1.

    References
    ----------
    Schadner, W. "Direct Fit for SVI Implied Volatilities", Journal of
    Derivatives (forthcoming). Implementation based on wol-fi/directSVI.
    """
    x = np.asarray(k, dtype=np.float64)
    y = np.asarray(w, dtype=np.float64)

    # Design matrices: D2 = [x², y²], D1 = [xy, x, y, 1]
    D2 = np.column_stack([x**2, y**2])
    D1 = np.column_stack([x * y, x, y, np.ones_like(x)])

    # Scatter matrices
    S22 = D2.T @ D2
    S21 = D2.T @ D1
    S11 = D1.T @ D1

    # Constraint matrix for hyperbola: z2² - 4*z0*z1 > 0
    # C1 encodes the quadratic form on the [z0, z1] block
    C1 = np.array([[0.0, -2.0], [-2.0, 0.0]])

    # Solve generalised eigenvalue problem: M @ a2 = lambda * C1 @ a2
    # where M = S22 - S21 @ inv(S11) @ S21.T
    S11_inv = np.linalg.inv(S11)
    M = S22 - S21 @ S11_inv @ S21.T

    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(C1) @ M)

    # Select eigenvector for smallest positive eigenvalue
    real_mask = np.isreal(eigvals)
    eigvals_real = np.real(eigvals)
    pos_mask = real_mask & (eigvals_real > 0)
    if not np.any(pos_mask):
        # Fallback: use eigenvector with smallest absolute eigenvalue
        idx = np.argmin(np.abs(eigvals_real))
    else:
        idx = np.where(pos_mask)[0][np.argmin(eigvals_real[pos_mask])]

    a2 = np.real(eigvecs[:, idx])
    a1 = -S11_inv @ S21.T @ a2

    z = np.concatenate([a2, a1])  # [z0, z1, z2, z3, z4, z5]

    # Normalise so z[1] = 1 (coefficient of y²)
    if abs(z[1]) > 1e-15:
        z = z / z[1]

    return z


def directsvi_total_variance(
    k: NDArray[np.float64],
    z0: float, z1: float, z2: float, z3: float, z4: float, z5: float,
) -> NDArray[np.float64]:
    """Evaluate the DirectSVI conic for given log-moneyness.

    Solves the conic z₀x² + z₁y² + z₂xy + z₃x + z₄y + z₅ = 0 for y
    via the quadratic formula::

        y = (−(z₂x + z₄) + √((z₂x + z₄)² − 4z₁(z₀x² + z₃x + z₅))) / (2z₁)

    Selects the positive root (total variance must be non-negative).

    Parameters
    ----------
    k : array
        Log-moneyness.
    z0, z1, z2, z3, z4, z5 : float
        Conic coefficients.

    Returns
    -------
    array
        Total variance w(k).
    """
    return _kernels.resolve("directsvi_w")(_as_f64(k), z0, z1, z2, z3, z4, z5)


class DirectSVI(Parametrization):
    """Direct algebraic SVI fit via conic section [Schadner].

    Linearises the SVI total variance curve as a hyperbola::

        z₀k² + z₁w² + z₂kw + z₃k + z₄w + z₅ = 0

    and solves a constrained eigenvalue problem for the 6 conic
    coefficients — no iterative optimisation needed.

    The direct fit is fast and robust but does not support penalty-based
    arbitrage enforcement (NO_BUTTERFLY / NO_CALENDAR). Only
    ArbitrageFreedom.QUASI is meaningful.

    Rendered formulas: https://pysvi.readthedocs.io/en/latest/models/directsvi.html
    """

    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """Fit conic coefficients via direct eigenvalue solve.

        Parameters
        ----------
        k : array
            Log-moneyness.
        w_target : array
            Market total variances.

        Returns
        -------
        dict or None
            {'z0', 'z1', 'z2', 'z3', 'z4', 'z5'}.
        """
        if self.arbitrage_condition != ArbitrageFreedom.QUASI:
            logger.warning(
                "DirectSVI only supports QUASI arbitrage condition; "
                "NO_BUTTERFLY / NO_CALENDAR flags are ignored."
            )

        z = directsvi_fit(k, w_target)
        return {
            "z0": float(z[0]),
            "z1": float(z[1]),
            "z2": float(z[2]),
            "z3": float(z[3]),
            "z4": float(z[4]),
            "z5": float(z[5]),
        }

    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """Evaluate w(k) from conic coefficients."""
        return directsvi_total_variance(
            k, params["z0"], params["z1"], params["z2"],
            params["z3"], params["z4"], params["z5"],
        )


class SABR(Parametrization):
    """SABR stochastic volatility model [Hagan et al. 2002].

    ::

        dF = α F^β dW₁,  dα = ν α dW₂,  d<W₁,W₂> = ρ dt

    Implied vols come from the Hagan (HKLW) lognormal asymptotic
    expansion; total variance is w(k) = σ_B(k)² T. The market standard
    for interest-rate (swaption/cap) and FX volatility smiles.

    β is fixed by market convention (not fitted): 1 for FX/equity,
    ~0.5 for rates, 0 for normal dynamics. Calibrates (α, ρ, ν) given
    β, F, T via L-BFGS-B with Nelder-Mead fallback.

    Butterfly-density derivatives are evaluated by finite differences
    (the Hagan expansion has no tractable closed-form w'', and the
    expansion itself can violate no-arbitrage for extreme strikes or
    long maturities — NO_BUTTERFLY is a numerical check, not a
    structural guarantee).

    Rendered formulas: https://pysvi.readthedocs.io/en/latest/models/sabr.html
    """

    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """Fit (α, ρ, ν) minimizing MSE(w_model, w_target) for fixed β.

        Parameters
        ----------
        k : NDArray[np.float64]
            Log-moneyness log(K/F).
        w_target : NDArray[np.float64]
            Market total variances σ_mkt² T.
        **kwargs
            Must contain 'T' (years) and 'F' (forward price).
            Optional 'beta': CEV exponent in [0, 1], default 0.5.
            Optional 'w_prev': prior slice total variance for calendar arb.

        Returns
        -------
        Dict[str, float] or None
            {'alpha', 'beta', 'rho', 'nu', 'F', 'T'} or None.
        """
        T = float(kwargs["T"])
        F = float(kwargs["F"])
        beta = float(kwargs.get("beta", 0.5))
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"SABR beta must be in [0, 1], got {beta}")
        if T <= 0 or F <= 0:
            raise ValueError(f"SABR requires T > 0 and F > 0, got T={T}, F={F}")


        k, w_target, k_grid, w_prev_arr, has_prev, check_butterfly, check_calendar = (
            _prepare_objective_inputs(k, w_target, self.arbitrage_condition, kwargs)
        )
        mode, loss_code, weights, w_lo, w_hi = _prepare_loss_inputs(k, w_target, kwargs)
        core = _kernels.resolve("sabr_obj")

        def objective(params):
            return core(
                np.asarray(params, dtype=np.float64), k, w_target, beta, F, T,
                k_grid, w_prev_arr, check_butterfly, check_calendar, has_prev,
                mode, weights, w_lo, w_hi, loss_code, f_scale,
            )

        init = _initialization(kwargs)
        # Initial guess: ATM vol maps to alpha via sigma_ATM ~ alpha / F^(1-beta)
        w_atm = float(np.interp(0.0, k, w_target))
        sigma_atm = np.sqrt(max(w_atm, 1e-12) / T)
        alpha0 = sigma_atm * F ** (1.0 - beta)
        x0 = np.array([max(alpha0, 1e-4), 0.0, 0.5])

        bounds = [
            (1e-8, None),      # alpha > 0
            (-0.999, 0.999),   # |rho| < 1
            (0.0, None),       # nu >= 0
        ]

        f_scale = _resolve_f_scale(
            kwargs, k,
            sabr_total_variance(k, x0[0], beta, x0[1], x0[2], F, T),
            w_target, mode, loss_code, weights, w_lo, w_hi,
            pilot=lambda: self._pilot_f_scale(
                k, w_target, kwargs, mode, weights, w_lo, w_hi
            ),
        )
        starts = _multistart_variants(x0, 1, 2) if init == "multi_start" else [x0]
        # Tight ftol/gtol: total-variance MSEs are O(1e-8) even mid-fit, so
        # scipy's default relative ftol would declare convergence too early.
        res = _minimize_with_starts(
            objective, starts, bounds,
            lbfgs_options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000},
            nm_options={"maxiter": 2000, "fatol": 1e-14, "xatol": 1e-10},
        )
        if res is None:
            return None

        alpha, rho, nu = res.x
        if alpha <= 0 or abs(rho) >= 0.999 or nu < 0:
            return None

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "rho": float(rho),
            "nu": float(nu),
            "F": float(F),
            "T": float(T),
        }

    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """Evaluate w(k) = σ_B(k)² T via the Hagan expansion."""
        return sabr_total_variance(
            k, params["alpha"], params["beta"], params["rho"],
            params["nu"], params["F"], params["T"],
        )
