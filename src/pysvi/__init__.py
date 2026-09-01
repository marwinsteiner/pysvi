"""PySVI: Volatility surface parametrizations."""

try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import version
    __version__ = version("svi-py")

from .models import (SVI, NaturalSVI, SSVI, ESSVI, JumpWings, DirectSVI, SABR,
                     ArbitrageFreedom,
                     svi_total_variance, ssvi_total_variance, essvi_total_variance,
                     jw_total_variance, directsvi_total_variance,
                     sabr_total_variance, sabr_implied_vol,
                     natural_total_variance, natural_to_raw, raw_to_natural,
                     use_numba, numba_available)
from .calibration import (prepare_slice, calibrate_slice, apply_slice,
                         calculate_implied_forward, get_model)
from .diagnostics import (check_slice_arbitrage, check_arbitrage,
                          SliceArbitrageReport, ArbitrageReport, LEE_BOUND)
from .surface import VolSurface, calibrate_surface

__all__ = [
    "SVI", "NaturalSVI", "SSVI", "ESSVI", "JumpWings", "DirectSVI", "SABR",
    "ArbitrageFreedom", "get_model",
    "svi_total_variance", "ssvi_total_variance", "essvi_total_variance",
    "jw_total_variance", "directsvi_total_variance",
    "sabr_total_variance", "sabr_implied_vol",
    "natural_total_variance", "natural_to_raw", "raw_to_natural",
    "use_numba", "numba_available",
    "check_slice_arbitrage", "check_arbitrage",
    "SliceArbitrageReport", "ArbitrageReport", "LEE_BOUND",
    "VolSurface", "calibrate_surface",
    "prepare_slice", "calibrate_slice", "apply_slice", "calculate_implied_forward"
]