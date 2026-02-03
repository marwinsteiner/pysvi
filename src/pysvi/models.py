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
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# @nb.njit(fastmath=True)
def ssvi_total_variance(
    k: np.ndarray,
    theta: float,
    rho: float,
    phi_theta: float
) -> np.ndarray:
    """SSVI total variance w(k)."""
    term1 = 1.0 + rho * phi_theta * k
    term2 = np.sqrt((phi_theta * k + rho)**2 + (1.0 - rho**2))
    return 0.5 * theta * (term1 + term2)

# @nb.njit(fastmath=True)
def essvi_total_variance(
    k: np.ndarray,
    theta: float,
    rho_theta: float,
    phi_theta: float
) -> np.ndarray:
    """eSSVI total variance w(k)."""
    inside = (phi_theta * k + rho_theta)**2 + (1.0 - rho_theta**2)
    term1 = 1.0 + rho_theta * phi_theta * k
    term2 = np.sqrt(inside)
    return 0.5 * theta * (term1 + term2)