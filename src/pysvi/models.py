# src/pysvi/models.py
"""
Core parametrizations for stochastic volatility inspired IV surfaces.
Pure NumPy/Numba functions for SVI, SSVI, eSSVI total variance.
Extensible via Parametrization ABC.
"""

# import numba as nb
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional
from numpy.typing import NDArray


# @nb.njit(fastmath=True)
def svi_total_variance(
        k: np.ndarray,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float
) -> np.ndarray:
    """Raw SVI total variance w(k)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


# @nb.njit(fastmath=True)
def ssvi_total_variance(
        k: np.ndarray,
        theta: float,
        rho: float,
        phi_theta: float
) -> np.ndarray:
    """SSVI total variance w(k)."""
    term1 = 1.0 + rho * phi_theta * k
    term2 = np.sqrt((phi_theta * k + rho) ** 2 + (1.0 - rho ** 2))
    return 0.5 * theta * (term1 + term2)


# @nb.njit(fastmath=True)
def essvi_total_variance(
        k: np.ndarray,
        theta: float,
        rho_theta: float,
        phi_theta: float
) -> np.ndarray:
    """eSSVI total variance w(k)."""
    inside = (phi_theta * k + rho_theta) ** 2 + (1.0 - rho_theta ** 2)
    term1 = 1.0 + rho_theta * phi_theta * k
    term2 = np.sqrt(inside)
    return 0.5 * theta * (term1 + term2)


class Parametrization(ABC):
    """Base class for IV surface parametrizations."""

    @abstractmethod
    def calibrate(
            self,
            k: NDArray[np.float64],
            w_target: NDArray[np.float64],
            **kwargs
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
            self,
            k: NDArray[np.float64],
            params: Dict[str, float]
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
            self,
            k: NDArray[np.float64],
            w_target: NDArray[np.float64],
            **kwargs
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

        def objective(params):
            a, b, rho, m, sigma = params
            penalty = 0.0
            if b <= 0: penalty += 1e6 * (1 - b) ** 2
            if abs(rho) >= 0.999: penalty += 1e6 * (abs(rho) - 0.999) ** 2
            if sigma <= 0: penalty += 1e6 * (1 - sigma) ** 2
            w_model = svi_total_variance(k, a, b, rho, m, sigma)
            mse = float(np.mean((w_target - w_model) ** 2))
            return mse + penalty

        a0 = float(np.nanmin(w_target))
        spread = float(np.nanmax(w_target) - a0)
        k_abs_max = float(np.max(np.abs(k)))
        denom = max(k_abs_max, 1.0)
        b0 = max(spread / denom, 1e-4)
        x0 = np.array([a0, b0, 0.0, float(np.median(k)), max(float(np.std(k)), 0.1)])

        bounds = [(None, None), (1e-8, None), (-0.999, 0.999), (None, None), (1e-8, None)]

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            # Fallback Nelder-Mead
            res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 2000})
            if not res.success:
                return None

        a, b, rho, m, sigma = res.x
        if b <= 0 or sigma <= 0 or abs(rho) >= 0.999:
            return None

        return {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(sigma)}

    def total_variance(
            self,
            k: NDArray[np.float64],
            params: Dict[str, float]
    ) -> NDArray[np.float64]:
        """Evaluate w(k) = a + b{ρ(k-m) + sqrt[(k-m)² + σ²]}."""
        return svi_total_variance(k, **params)


class SSVI(Parametrization):
    """Surface-consistent SSVI [Gatheral & Jacquier 2014].

    w(k;θ) = θ/2 [1 + ρ φ(θ) k + sqrt{(φ(θ) k + ρ)² + (1-ρ²)}]

    θ = ATM total variance (fixed per slice, typically σ_ATM² T)
    φ(θ) = η / sqrt(θ) - curvature scale independent of ATM level

    Guarantees no butterfly arbitrage across strikes for fixed θ.
    Calibrates only ρ, η (2 params) given θ.
    """

    def calibrate(
            self,
            k: NDArray[np.float64],
            w_target: NDArray[np.float64],
            **kwargs
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

        def objective(params):
            rho, eta = params
            penalty = 0.0
            if abs(rho) >= 0.999: penalty += 1e6 * (abs(rho) - 0.999) ** 2
            if eta <= 0: penalty += 1e6 * (1 - eta) ** 2
            phi_theta = eta / np.sqrt(theta)
            w_model = ssvi_total_variance(k, theta, rho, phi_theta)
            mse = float(np.mean((w_target - w_model) ** 2))
            return mse + penalty

        x0 = np.array([0.0, 1.0])
        bounds = [(-0.999, 0.999), (1e-8, None)]

        res = minimize(objective, x0, args=(float(theta),), method="L-BFGS-B", bounds=bounds)
        if not res.success:
            res = minimize(objective, x0, args=(float(theta),), method="Nelder-Mead")
            if not res.success:
                return None

        rho, eta = res.x
        if eta <= 0 or abs(rho) >= 0.999:
            return None

        return {"rho": float(rho), "eta": float(eta), "theta": float(theta)}

    def total_variance(
            self,
            k: NDArray[np.float64],
            params: Dict[str, float]
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
            self,
            k: NDArray[np.float64],
            w_target: NDArray[np.float64],
            **kwargs
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
        theta_ref= kwargs["theta_ref"]

        from scipy.optimize import minimize

        if theta_ref is None:
            theta_ref = theta

        def objective(params):
            rho0, rho1, alpha, eta = params
            penalty = 0.0
            if eta <= 0: penalty += 1e6 * (1 - eta) ** 2
            theta_ratio = theta / max(theta_ref, 1e-12)
            rho_theta = np.clip(rho0 + rho1 * (theta_ratio ** alpha), -0.999, 0.999)
            phi_theta = eta / np.sqrt(theta)
            w_model = essvi_total_variance(k, theta, rho_theta, phi_theta)
            mse = float(np.mean((w_target - w_model) ** 2))
            penalty += 1e2 * max(0.0, abs(rho_theta) - 0.95)
            return mse + penalty

        x0 = np.array([0.0, -0.5, 0.5, 1.0])
        bounds = [(-0.999, 0.999), (-2.0, 2.0), (-2.0, 2.0), (1e-8, None)]

        res = minimize(objective, x0, args=(float(theta), float(theta_ref)), method="L-BFGS-B", bounds=bounds)
        if not res.success:
            res = minimize(objective, x0, args=(float(theta), float(theta_ref)), method="Nelder-Mead")
            if not res.success:
                return None

        rho0, rho1, alpha, eta = res.x
        if eta <= 0:
            return None

        theta_ratio = theta / max(theta_ref, 1e-12)
        rho_theta = np.clip(rho0 + rho1 * (theta_ratio ** alpha), -0.999, 0.999)

        return {
            "rho0": float(rho0), "rho1": float(rho1), "alpha": float(alpha),
            "eta": float(eta), "theta": float(theta), "theta_ref": float(theta_ref),
            "rho_theta": float(rho_theta)
        }

    def total_variance(
            self,
            k: NDArray[np.float64],
            params: Dict[str, float]
    ) -> NDArray[np.float64]:
        theta = params["theta"]
        rho_theta = params.get("rho_theta", params["rho0"] + params["rho1"] *
                               (theta / max(params["theta_ref"], 1e-12)) ** params["alpha"])
        rho_theta = np.clip(rho_theta, -0.999, 0.999)
        phi_theta = params["eta"] / np.sqrt(theta)
        return essvi_total_variance(k, theta, rho_theta, phi_theta)
