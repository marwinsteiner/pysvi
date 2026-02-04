import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# src/pysvi absolute imports (tests/ and src/ siblings)
from src.pysvi.models import (
    SVI, SSVI, ESSVI,
    svi_total_variance
)
from src.pysvi.calibration import (
    prepare_slice, calibrate_slice, apply_slice, get_model
)

from typing import cast


@pytest.fixture
def atm_slice() -> pd.DataFrame:
    """Synthetic realistic slice: F=100, T=0.25, noisy SVI smile."""
    np.random.seed(42)
    F, T = 100.0, 0.25
    k_true = np.linspace(-0.2, 0.2, 21)
    strikes = F * np.exp(k_true)

    # Ground truth SVI params
    true_params = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}
    w_true = svi_total_variance(k_true, **true_params)
    iv_true = np.sqrt(w_true / T)

    # Add realistic market noise (±2bps)
    iv_mkt = iv_true + 0.0002 * np.random.randn(21)

    return pd.DataFrame({
        "strike": strikes,
        "iv": iv_mkt,
        "maturity": T,
        "implied_forward": F
    })


def _valid_slice_for_prepare() -> pd.DataFrame:
    """Factory: not a fixture, can be called directly."""
    np.random.seed(42)
    F, T = 100.0, 0.25
    strikes = F * np.exp(np.linspace(-0.15, 0.15, 15))
    ivs = 0.20 + 0.05 * np.abs(np.log(strikes / F)) + 0.001 * np.random.randn(15)
    return pd.DataFrame({
        "strike": strikes,
        "iv": ivs,
        "maturity": T,
        "implied_forward": F
    })


def _invalid_slice_for_prepare() -> pd.DataFrame:
    """Factory: invalid data for testing."""
    return pd.DataFrame({
        "strike": [0.0, np.nan, -5.0],
        "iv": [-0.1, 0.0, np.inf],
        "maturity": 0.25,
        "implied_forward": 100.0
    })


@pytest.fixture
def valid_slice_for_prepare():
    return _valid_slice_for_prepare()


@pytest.fixture
def invalid_slice_for_prepare():
    return _invalid_slice_for_prepare()


@pytest.fixture(params=[_valid_slice_for_prepare(), _invalid_slice_for_prepare()])
def prepare_slices(request) -> pd.DataFrame:
    return request.param


@pytest.fixture
def fitted_slice_for_apply(calibrated_svi) -> Tuple[pd.DataFrame, Dict[str, float], SVI]:
    """Pre-calibrated slice ready for apply_slice."""
    model, df_slice, params = calibrated_svi
    return df_slice, params, model


@pytest.fixture
def calibrated_svi(atm_slice) -> Tuple[SVI, pd.DataFrame, Dict[str, float]]:
    """SVI + slice + guaranteed calibration."""
    df_slice = atm_slice
    model = cast(SVI, get_model("svi"))  # ← mypy cast
    params = calibrate_slice(df_slice, model)
    assert params is not None
    assert all(params[k] > 0 for k in ["a", "b", "sigma"])
    assert abs(params["rho"]) < 0.999
    return model, df_slice, params


@pytest.fixture
def ssvi_calibrated(atm_slice) -> Tuple[SSVI, pd.DataFrame, Dict[str, float]]:
    """SSVI + explicit theta calibration."""
    df_slice = atm_slice
    model = cast(SSVI, get_model("ssvi"))  # ← mypy cast
    theta = float(np.nanmin(df_slice["iv"] ** 2 * df_slice["maturity"]))
    params = calibrate_slice(df_slice, model, theta=theta)
    assert params is not None
    assert params["theta"] == theta
    assert params["eta"] > 0 and abs(params["rho"]) < 0.999
    return model, df_slice, params


@pytest.fixture
def essvi_calibrated(atm_slice) -> Tuple[ESSVI, pd.DataFrame, Dict[str, float]]:
    """eSSVI + explicit theta/theta_ref calibration."""
    df_slice = atm_slice
    model = cast(ESSVI, get_model("essvi"))
    theta = float(np.nanmin(df_slice["iv"] ** 2 * df_slice["maturity"]))  # ATM w
    theta_ref = theta  # Self-reference for single slice
    params = calibrate_slice(df_slice, model, theta=theta, theta_ref=theta_ref)
    assert params is not None
    assert params["theta"] == theta
    assert params["eta"] > 0
    assert abs(params["rho_theta"]) < 0.999
    return model, df_slice, params
