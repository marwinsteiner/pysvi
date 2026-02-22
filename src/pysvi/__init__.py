"""PySVI: Volatility surface parametrizations."""

try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import version
    __version__ = version("svi-py")

from .models import (SVI, SSVI, ESSVI, JumpWings, ArbitrageFreedom,
                     svi_total_variance, ssvi_total_variance, essvi_total_variance,
                     jw_total_variance)
from .calibration import (prepare_slice, calibrate_slice, apply_slice,
                         calculate_implied_forward, get_model)

__all__ = [
    "SVI", "SSVI", "ESSVI", "JumpWings", "ArbitrageFreedom", "get_model",
    "svi_total_variance", "ssvi_total_variance", "essvi_total_variance",
    "jw_total_variance",
    "prepare_slice", "calibrate_slice", "apply_slice", "calculate_implied_forward"
]