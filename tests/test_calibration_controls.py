"""Calibration controls: objectives, robust losses, initialisation."""

import numpy as np
import pytest

from src.pysvi.models import (
    SVI, NaturalSVI, SSVI, SABR, svi_total_variance,
)
from src.pysvi import _kernels as K

TRUE = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}
_K_GRID = np.linspace(-0.4, 0.4, 41)
_W_TRUE = svi_total_variance(_K_GRID, **TRUE)


def _fit_rmse(params, w_ref=_W_TRUE, k=_K_GRID):
    w_fit = svi_total_variance(k, **{p: params[p] for p in ("a", "b", "rho", "m", "sigma")})
    return float(np.sqrt(np.mean((w_ref - w_fit) ** 2)))


# ── Objectives ───────────────────────────────────────────────────────

@pytest.mark.parametrize("objective", ["total_variance", "implied_vol", "price", "vega_weighted"])
def test_objectives_recover_clean_smile(objective):
    """Every residual space recovers a clean smile to tiny w-RMSE.

    multi_start decouples objective-space correctness from the raw-SVI
    landscape's local minima (the default start can land in a poor basin).
    """
    params = SVI().calibrate(
        _K_GRID, _W_TRUE, objective=objective, initialization="multi_start"
    )
    assert params is not None
    assert _fit_rmse(params) < 1e-5, objective


def test_black_call_matches_py_vollib():
    """The price-space kernel matches py_vollib's Black formula."""
    from py_vollib.black import black
    T = 0.25
    for k_i, w in [(-0.2, 0.012), (0.0, 0.01), (0.15, 0.011), (0.4, 0.02)]:
        expected = black("c", 1.0, float(np.exp(k_i)), T, 0.0, float(np.sqrt(w / T)))
        got = K._PLAIN["black_call"](k_i, w)
        np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_bid_ask_objective_stays_inside_band(atm_slice):
    """bid_ask calibration lands inside the quoted band where feasible."""
    noisy = _W_TRUE * (1.0 + 0.02 * np.sin(37.0 * _K_GRID))  # noisy mids
    w_bid = _W_TRUE - 5e-4
    w_ask = _W_TRUE + 5e-4
    params = SVI().calibrate(
        _K_GRID, noisy, objective="bid_ask", w_bid=w_bid, w_ask=w_ask
    )
    assert params is not None
    w_fit = svi_total_variance(_K_GRID, **params)
    violation = np.maximum(w_bid - w_fit, 0.0) + np.maximum(w_fit - w_ask, 0.0)
    assert float(violation.max()) < 1e-6, f"band violation {violation.max():.2e}"


def test_unknown_objective_and_loss_raise():
    with pytest.raises(ValueError, match="unknown objective"):
        SVI().calibrate(_K_GRID, _W_TRUE, objective="prices")
    with pytest.raises(ValueError, match="unknown loss"):
        SVI().calibrate(_K_GRID, _W_TRUE, loss="l1")
    with pytest.raises(ValueError, match="w_bid"):
        SVI().calibrate(_K_GRID, _W_TRUE, objective="bid_ask")


# ── Robust losses ────────────────────────────────────────────────────

def test_robust_loss_resists_corrupted_wing():
    """One corrupted far-wing quote distorts l2 far more than soft_l1."""
    w_bad = _W_TRUE.copy()
    w_bad[-1] *= 1.5  # single bad call-wing quote
    clean = slice(0, -1)

    p_l2 = SVI().calibrate(_K_GRID, w_bad)
    p_robust = SVI().calibrate(_K_GRID, w_bad, loss="soft_l1")
    assert p_l2 is not None and p_robust is not None

    rmse_l2 = _fit_rmse(p_l2, _W_TRUE[clean], _K_GRID[clean])
    rmse_robust = _fit_rmse(p_robust, _W_TRUE[clean], _K_GRID[clean])
    assert rmse_robust < rmse_l2, (rmse_robust, rmse_l2)
    assert rmse_robust < 1e-4


@pytest.mark.parametrize("loss", ["huber", "soft_l1", "cauchy"])
def test_robust_losses_recover_clean_smile(loss):
    """Robust losses reproduce the l2 fit on clean data (multi_start
    decouples the check from raw SVI's local-minimum landscape)."""
    params = SVI().calibrate(
        _K_GRID, _W_TRUE, loss=loss, initialization="multi_start"
    )
    assert params is not None
    assert _fit_rmse(params) < 1e-4, loss


def test_explicit_f_scale_accepted():
    params = SVI().calibrate(
        _K_GRID, _W_TRUE, loss="huber", f_scale=1e-4,
        initialization="multi_start",
    )
    assert params is not None
    assert _fit_rmse(params) < 1e-4


# ── Initialisation ───────────────────────────────────────────────────

def test_multi_start_deterministic():
    """Two multi_start runs give bitwise-identical parameters."""
    p1 = SVI().calibrate(_K_GRID, _W_TRUE, initialization="multi_start")
    p2 = SVI().calibrate(_K_GRID, _W_TRUE, initialization="multi_start")
    assert p1 is not None
    assert p1 == p2


def test_multi_start_escapes_bad_basin():
    """Regression: the default start converges to a poor local optimum on
    this clean smile under tight tolerances; multi_start recovers the
    true fit (review-issue acceptance case for pysvi#9)."""
    p_default = SVI().calibrate(_K_GRID, _W_TRUE)
    p_multi = SVI().calibrate(_K_GRID, _W_TRUE, initialization="multi_start")
    assert p_default is not None and p_multi is not None
    assert _fit_rmse(p_multi) < 1e-5
    assert _fit_rmse(p_multi) < _fit_rmse(p_default) / 10.0


def test_multi_start_never_worse_than_default():
    """The default start is in the start set, so multi_start cannot lose."""
    rng = np.random.default_rng(7)
    w_noisy = _W_TRUE + 2e-5 * rng.standard_normal(_K_GRID.size)
    p_default = SVI().calibrate(_K_GRID, w_noisy)
    p_multi = SVI().calibrate(_K_GRID, w_noisy, initialization="multi_start")
    assert p_default is not None and p_multi is not None
    mse_default = float(np.mean((w_noisy - svi_total_variance(_K_GRID, **p_default)) ** 2))
    mse_multi = float(np.mean((w_noisy - svi_total_variance(_K_GRID, **p_multi)) ** 2))
    assert mse_multi <= mse_default + 1e-14


def test_multi_start_other_models(atm_slice):
    """multi_start runs for SSVI and SABR."""
    from src.pysvi.calibration import prepare_slice
    k, w_target, F = prepare_slice(atm_slice)
    T = float(atm_slice["maturity"].iloc[0])
    theta = float(np.nanmin(atm_slice["iv"] ** 2 * atm_slice["maturity"]))
    p_ssvi = SSVI().calibrate(k, w_target, theta=theta, initialization="multi_start")
    assert p_ssvi is not None
    p_sabr = SABR().calibrate(k, w_target, T=T, F=F, beta=1.0,
                              initialization="multi_start")
    assert p_sabr is not None


def test_jump_wings_initialization():
    """Data-driven wing readoff works for SVI and NaturalSVI."""
    p_svi = SVI().calibrate(_K_GRID, _W_TRUE, initialization="jump_wings")
    assert p_svi is not None
    assert _fit_rmse(p_svi) < 1e-5

    model = NaturalSVI()
    p_nat = model.calibrate(_K_GRID, _W_TRUE, initialization="jump_wings")
    assert p_nat is not None
    w_fit = model.total_variance(_K_GRID, p_nat)
    assert float(np.sqrt(np.mean((_W_TRUE - w_fit) ** 2))) < 1e-5


def test_jump_wings_rejected_elsewhere():
    with pytest.raises(ValueError, match="jump_wings"):
        SSVI().calibrate(_K_GRID, _W_TRUE, theta=0.01, initialization="jump_wings")


def test_unknown_initialization_raises():
    with pytest.raises(ValueError, match="unknown initialization"):
        SVI().calibrate(_K_GRID, _W_TRUE, initialization="random")


# ── Controls compose with the numba backend ──────────────────────────

def test_controls_backend_parity(backend_mode):
    """Objectives and losses run identically under both backends."""
    params = SVI().calibrate(
        _K_GRID, _W_TRUE, objective="vega_weighted", loss="soft_l1",
        initialization="multi_start",
    )
    assert params is not None
    assert _fit_rmse(params) < 1e-4
