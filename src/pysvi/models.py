# src/pysvi/models.py
"""
Core parametrizations for stochastic volatility inspired IV surfaces.
Pure NumPy/Numba functions for SVI, SSVI, eSSVI total variance.
Extensible via Parametrization ABC.
"""

# import numba as nb
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
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
    """Raw SVI parametrization."""

    def calibrate(
            self,
            k: NDArray[np.float64],
            w_target: NDArray[np.float64],
            **kwargs
    ) -> Optional[Dict[str, float]]:
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
        return svi_total_variance(k, **params)
