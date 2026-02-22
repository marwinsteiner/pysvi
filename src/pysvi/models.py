# src/pysvi/models.py
"""
Core parametrizations for stochastic volatility inspired IV surfaces.
Pure NumPy/Numba functions for SVI, SSVI, eSSVI total variance.
Extensible via Parametrization ABC.
"""

# import numba as nb
import numpy as np
from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Dict, Optional
from numpy.typing import NDArray


class ArbitrageFreedom(Flag):
    """Configurable arbitrage-freeness constraints for IV parametrizations.

    Combine flags with ``|`` to enforce multiple conditions simultaneously.

    Attributes
    ----------
    QUASI : default
        Soft parameter-bound constraints only (b > 0, |rho| < 1, sigma > 0).
    NO_BUTTERFLY : flag
        Enforce non-negative density g(k) >= 0 across strikes (no static arb).
    NO_CALENDAR : flag
        Enforce non-decreasing total variance in maturity (no calendar spread arb).
    """

    QUASI = 0
    NO_BUTTERFLY = auto()
    NO_CALENDAR = auto()


# @nb.njit(fastmath=True)
def svi_total_variance(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray:
    """Raw SVI total variance w(k)."""
    z = k - m
    return a + b * (rho * z + np.sqrt(z * z + sigma * sigma))


# @nb.njit(fastmath=True)
def ssvi_total_variance(
    k: np.ndarray, theta: float, rho: float, phi_theta: float
) -> np.ndarray:
    """SSVI total variance w(k)."""
    term1 = 1.0 + rho * phi_theta * k
    term2 = np.sqrt((phi_theta * k + rho) ** 2 + (1.0 - rho**2))
    return 0.5 * theta * (term1 + term2)


# @nb.njit(fastmath=True)
def essvi_total_variance(
    k: np.ndarray, theta: float, rho_theta: float, phi_theta: float
) -> np.ndarray:
    """eSSVI total variance w(k)."""
    inside = (phi_theta * k + rho_theta) ** 2 + (1.0 - rho_theta**2)
    term1 = 1.0 + rho_theta * phi_theta * k
    term2 = np.sqrt(inside)
    return 0.5 * theta * (term1 + term2)


def _butterfly_penalty(
    k: np.ndarray, w: np.ndarray, dw: np.ndarray, d2w: np.ndarray
) -> float:
    """Penalty for butterfly arbitrage violations.

    The call price density is proportional to g(k) where

        g(k) = (1 - k w'/(2w))^2 - (w')^2/4 (1/w + 1/4) + w''/2

    Butterfly arbitrage is absent iff g(k) >= 0 for all k.
    Returns a penalty proportional to the integral of max(-g, 0).
    """
    g = (1.0 - k * dw / (2.0 * w)) ** 2 - (dw**2) / 4.0 * (1.0 / w + 0.25) + d2w / 2.0
    violations = np.maximum(-g, 0.0)
    return float(np.sum(violations**2))


def _svi_derivatives(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute w, w', w'' for raw SVI parametrization."""
    z = k - m
    r = np.sqrt(z * z + sigma * sigma)
    w = a + b * (rho * z + r)
    dw = b * (rho + z / r)
    d2w = b * sigma**2 / r**3
    return w, dw, d2w


def _ssvi_derivatives(
    k: np.ndarray, theta: float, rho: float, phi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute w, w', w'' for SSVI/eSSVI parametrization."""
    u = phi * k + rho
    disc = np.sqrt(u**2 + 1.0 - rho**2)
    w = 0.5 * theta * (1.0 + rho * phi * k + disc)
    dw = 0.5 * theta * phi * (rho + u / disc)
    d2w = 0.5 * theta * phi**2 * (1.0 - rho**2) / disc**3
    return w, dw, d2w


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
    diff = w_prev - w_current  # positive where calendar arb exists
    violations = np.maximum(diff, 0.0)
    return float(np.sum(violations**2))


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
            Extra model-specific arguments (e.g. theta for SSVI/eSSVI).

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


class SVI(Parametrization):
    """Raw SVI total variance parametrization [Gatheral 2004].

    w(k) = a + b {ρ(k-m) + sqrt[(k-m)² + σ²]}

    No-arbitrage constraints softly enforced via bounds/penalties:
    * b > 0 (positive slope)
    * |ρ| < 1 (correlation)
    * σ > 0 (vol of vol)

    Calibrates via L-BFGS-B (bounded) → Nelder-Mead fallback.
    Initial guess: ATM a, median m, wing-informed b/σ.
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
        from scipy.optimize import minimize

        check_butterfly = ArbitrageFreedom.NO_BUTTERFLY in self.arbitrage_condition
        check_calendar = ArbitrageFreedom.NO_CALENDAR in self.arbitrage_condition
        need_grid = check_butterfly or check_calendar
        k_grid = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 200) if need_grid else None
        w_prev = kwargs.get("w_prev")

        def objective(params):
            a, b, rho, m, sigma = params
            penalty = 0.0
            if b <= 0:
                penalty += 1e6 * (1 - b) ** 2
            if abs(rho) >= 0.999:
                penalty += 1e6 * (abs(rho) - 0.999) ** 2
            if sigma <= 0:
                penalty += 1e6 * (1 - sigma) ** 2
            w_model = svi_total_variance(k, a, b, rho, m, sigma)
            mse = float(np.mean((w_target - w_model) ** 2))
            if need_grid and b > 0 and sigma > 0:
                w_g = svi_total_variance(k_grid, a, b, rho, m, sigma)
                if check_butterfly:
                    _, dw_g, d2w_g = _svi_derivatives(k_grid, a, b, rho, m, sigma)
                    penalty += 1e4 * _butterfly_penalty(k_grid, w_g, dw_g, d2w_g)
                if check_calendar and w_prev is not None:
                    penalty += 1e4 * _calendar_penalty(k_grid, w_g, w_prev)
            return mse + penalty

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

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            # Fallback Nelder-Mead
            res = minimize(
                objective, x0, method="Nelder-Mead", options={"maxiter": 2000}
            )
            if not res.success:
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


class SSVI(Parametrization):
    """Surface-consistent SSVI [Gatheral & Jacquier 2014].

    w(k;θ) = θ/2 [1 + ρ φ(θ) k + sqrt{(φ(θ) k + ρ)² + (1-ρ²)}]

    θ = ATM total variance (fixed per slice, typically σ_ATM² T)
    φ(θ) = η / sqrt(θ) - curvature scale independent of ATM level

    Guarantees no butterfly arbitrage across strikes for fixed θ.
    Calibrates only ρ, η (2 params) given θ.
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
        theta = kwargs["theta"]
        from scipy.optimize import minimize

        check_butterfly = ArbitrageFreedom.NO_BUTTERFLY in self.arbitrage_condition
        check_calendar = ArbitrageFreedom.NO_CALENDAR in self.arbitrage_condition
        need_grid = check_butterfly or check_calendar
        k_grid = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 200) if need_grid else None
        w_prev = kwargs.get("w_prev")

        def objective(params, k, w_target, theta):
            rho, eta = params
            penalty = 0.0
            if abs(rho) >= 0.999:
                penalty += 1e6 * (abs(rho) - 0.999) ** 2
            if eta <= 0:
                penalty += 1e6 * (1 - eta) ** 2
            phi_theta = eta / np.sqrt(theta)
            w_model = ssvi_total_variance(k, theta, rho, phi_theta)
            mse = float(np.mean((w_target - w_model) ** 2))
            if need_grid and eta > 0:
                w_g = ssvi_total_variance(k_grid, theta, rho, phi_theta)
                if check_butterfly:
                    _, dw_g, d2w_g = _ssvi_derivatives(k_grid, theta, rho, phi_theta)
                    penalty += 1e4 * _butterfly_penalty(k_grid, w_g, dw_g, d2w_g)
                if check_calendar and w_prev is not None:
                    penalty += 1e4 * _calendar_penalty(k_grid, w_g, w_prev)
            return mse + penalty

        x0 = np.array([0.0, 1.0])
        bounds = [(-0.999, 0.999), (1e-8, None)]

        res = minimize(objective, x0, args=(k, w_target, float(theta)), method="L-BFGS-B", bounds=bounds)
        if not res.success:
            res = minimize(objective, x0, args=(k, w_target, float(theta)), method="Nelder-Mead")
            if not res.success:
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


class ESSVI(Parametrization):
    """Extended SSVI with ρ(θ) parametrization.

    w(k;θ) = θ/2 [1 + ρ(θ) φ(θ) k + sqrt{(φ(θ) k + ρ(θ))² + (1-ρ(θ)²)}]

    ρ(θ) = clip(ρ₀ + ρ₁ (θ/θ_ref)^α, -0.999, 0.999)  ← term structure skew
    φ(θ) = η / sqrt(θ)                                ← curvature

    θ_ref smooths ρ across maturities (often median ATM θ). 4 params total.
    Enables realistic calendar skew evolution.
    """

    def calibrate(
        self, k: NDArray[np.float64], w_target: NDArray[np.float64], **kwargs
    ) -> Optional[Dict[str, float]]:
        """Fit ρ₀, ρ₁, α, η given θ, θ_ref via penalized MSE.

        Heavy penalty on η≤0, mild on |ρ(θ)|>0.95 for stability.

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
        theta = kwargs["theta"]
        theta_ref = kwargs["theta_ref"]

        from scipy.optimize import minimize

        if theta_ref is None:
            theta_ref = theta

        check_butterfly = ArbitrageFreedom.NO_BUTTERFLY in self.arbitrage_condition
        check_calendar = ArbitrageFreedom.NO_CALENDAR in self.arbitrage_condition
        need_grid = check_butterfly or check_calendar
        k_grid = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 200) if need_grid else None
        w_prev = kwargs.get("w_prev")

        def objective(params, k, w_target, theta, theta_ref):
            rho0, rho1, alpha, eta = params
            penalty = 0.0
            if eta <= 0:
                penalty += 1e6 * (1 - eta) ** 2
            theta_ratio = theta / max(theta_ref, 1e-12)
            rho_theta = np.clip(rho0 + rho1 * (theta_ratio**alpha), -0.999, 0.999)
            phi_theta = eta / np.sqrt(theta)
            w_model = essvi_total_variance(k, theta, rho_theta, phi_theta)
            mse = float(np.mean((w_target - w_model) ** 2))
            penalty += 1e2 * max(0.0, abs(rho_theta) - 0.95)
            if need_grid and eta > 0:
                w_g = essvi_total_variance(k_grid, theta, rho_theta, phi_theta)
                if check_butterfly:
                    _, dw_g, d2w_g = _ssvi_derivatives(k_grid, theta, rho_theta, phi_theta)
                    penalty += 1e4 * _butterfly_penalty(k_grid, w_g, dw_g, d2w_g)
                if check_calendar and w_prev is not None:
                    penalty += 1e4 * _calendar_penalty(k_grid, w_g, w_prev)
            return mse + penalty

        x0 = np.array([0.0, -0.5, 0.5, 1.0])
        bounds = [(-0.999, 0.999), (-2.0, 2.0), (-2.0, 2.0), (1e-8, None)]

        res = minimize(objective, x0, args=(k, w_target, float(theta), float(theta_ref)), method="L-BFGS-B", bounds=bounds)
        if not res.success:
            res = minimize(objective, x0, args=(k, w_target, float(theta), float(theta_ref)), method="Nelder-Mead")
            if not res.success:
                return None

        rho0, rho1, alpha, eta = res.x
        if eta <= 0:
            return None

        theta_ratio = theta / max(theta_ref, 1e-12)
        rho_theta = np.clip(rho0 + rho1 * (theta_ratio**alpha), -0.999, 0.999)

        return {
            "rho0": float(rho0),
            "rho1": float(rho1),
            "alpha": float(alpha),
            "eta": float(eta),
            "theta": float(theta),
            "theta_ref": float(theta_ref),
            "rho_theta": float(rho_theta),
        }

    def total_variance(
        self, k: NDArray[np.float64], params: Dict[str, float]
    ) -> NDArray[np.float64]:
        theta = params["theta"]
        rho_theta = params.get(
            "rho_theta",
            params["rho0"]
            + params["rho1"]
            * (theta / max(params["theta_ref"], 1e-12)) ** params["alpha"],
        )
        rho_theta = np.clip(rho_theta, -0.999, 0.999)
        phi_theta = params["eta"] / np.sqrt(theta)
        return essvi_total_variance(k, theta, rho_theta, phi_theta)
