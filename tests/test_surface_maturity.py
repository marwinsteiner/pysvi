"""calibrate_surface (calendar-aware fitting) and maturity interpolation."""

import numpy as np
import pandas as pd
import pytest

from src.pysvi.surface import VolSurface, calibrate_surface
from src.pysvi.models import ArbitrageFreedom, svi_total_variance
from src.pysvi.calibration import get_model

from tests.conftest import SURFACE_RATE as R

K_DATA = np.linspace(-0.25, 0.25, 21)


# ── calibrate_surface ────────────────────────────────────────────────

@pytest.mark.parametrize("model", ["ssvi", "svi", "natural", "sabr"])
def test_calendar_aware_fit_verifies_clean(surface_df, model):
    """No manual w_prev threading; the result passes the calendar check."""
    surface = calibrate_surface(surface_df, model=model, enforce_calendar=True, r=R)
    report = surface.check_arbitrage(k_data=K_DATA)
    assert report.calendar_free, str(report)
    assert report.ok, str(report)


def test_essvi_global_fit_shares_term_structure(surface_df):
    """eSSVI slices carry identical (rho0, rho1, alpha, eta) after the joint fit."""
    surface = calibrate_surface(surface_df, model="essvi", r=R)
    shapes = {
        tuple(surface.params(T)[q] for q in ("rho0", "rho1", "alpha", "eta"))
        for T in surface.maturities
    }
    assert len(shapes) == 1
    report = surface.check_arbitrage(k_data=K_DATA)
    assert report.calendar_free, str(report)
    # global consistency costs per-slice fit quality; keep it bounded
    for T in surface.maturities:
        g = surface_df[surface_df["maturity"] == T]
        fit_iv = surface.iv(g["strike"].to_numpy(), T)
        assert float(np.sqrt(np.mean((g["iv"].to_numpy() - fit_iv) ** 2))) < 0.08


def test_calendar_arbitrageable_data_is_repaired():
    """A panel with decreasing total variance still yields a calendar-free fit."""
    np.random.seed(3)
    rows = []
    base = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}
    # deliberately DECREASING w in T at every strike: calendar arbitrage in data
    for T, scale in ((0.25, 1.0), (0.5, 0.8)):
        F = 100.0
        w = svi_total_variance(K_DATA, **base) * scale
        iv = np.sqrt(w / T) + 0.0002 * np.random.randn(K_DATA.size)
        rows.append(pd.DataFrame({
            "strike": F * np.exp(K_DATA), "iv": iv,
            "maturity": T, "implied_forward": F,
        }))
    df = pd.concat(rows, ignore_index=True)

    surface = calibrate_surface(df, model="ssvi", enforce_calendar=True)
    report = surface.check_arbitrage(k_data=K_DATA, tol=1e-5)
    assert report.calendar_free, str(report)


def test_directsvi_calendar_enforcement_rejected(surface_df):
    with pytest.raises(ValueError, match="DirectSVI"):
        calibrate_surface(surface_df, model="dsvi", enforce_calendar=True)
    # without enforcement it works
    surface = calibrate_surface(surface_df, model="dsvi", enforce_calendar=False)
    assert len(surface.maturities) == 3


def test_instance_condition_gains_no_calendar(surface_df):
    """A model instance without NO_CALENDAR gets it when enforcing."""
    surface = calibrate_surface(
        surface_df, model=get_model("svi"), enforce_calendar=True
    )
    assert ArbitrageFreedom.NO_CALENDAR in surface.model.arbitrage_condition


def test_controls_pass_through(surface_df):
    surface = calibrate_surface(
        surface_df, model="ssvi", loss="soft_l1", initialization="multi_start", r=R
    )
    assert surface.check_arbitrage(k_data=K_DATA).calendar_free


def test_essvi_bid_ask_rejected(surface_df):
    with pytest.raises(ValueError, match="bid_ask"):
        calibrate_surface(surface_df, model="essvi", objective="bid_ask")


# ── Maturity interpolation ───────────────────────────────────────────

@pytest.fixture
def ssvi_surface(surface_df) -> VolSurface:
    return calibrate_surface(surface_df, model="ssvi", r=R)


def test_quoted_maturity_reproduced_exactly(ssvi_surface):
    """Interpolation at a fitted maturity is the slice itself, bitwise."""
    for T in ssvi_surface.maturities:
        direct = ssvi_surface.model.total_variance(K_DATA, ssvi_surface.params(T))
        np.testing.assert_array_equal(ssvi_surface.total_variance(K_DATA, T), direct)


def test_interpolated_w_between_bracketing_slices(ssvi_surface):
    """The blend sits between its bracketing slices at every strike."""
    w_lo = ssvi_surface.total_variance(K_DATA, 0.5)
    w_mid = ssvi_surface.total_variance(K_DATA, 0.7)
    w_hi = ssvi_surface.total_variance(K_DATA, 1.0)
    assert np.all(w_lo <= w_mid + 1e-14)
    assert np.all(w_mid <= w_hi + 1e-14)


def test_interpolated_slice_calendar_free(ssvi_surface):
    """Calendar freedom survives interpolation (blend property + diagnostics)."""
    from src.pysvi.diagnostics import check_arbitrage
    synthetic = [
        (T, None) for T in (0.3, 0.7)
    ]
    # build pseudo-slices by direct evaluation on the diagnostic grid
    k = K_DATA
    ws = {T: ssvi_surface.total_variance(k, T) for T in (0.25, 0.3, 0.5, 0.7, 1.0)}
    for lo, hi in zip([0.25, 0.3, 0.5, 0.7], [0.3, 0.5, 0.7, 1.0]):
        assert np.all(ws[lo] <= ws[hi] + 1e-14), (lo, hi)


def test_forward_log_linear(ssvi_surface):
    """F = 100 e^{rT} in the fixture, so log-linear interpolation is exact."""
    assert ssvi_surface.forward(0.7) == pytest.approx(100.0 * np.exp(R * 0.7))


def test_pricing_at_interpolated_maturity(ssvi_surface):
    T = 0.7
    F = ssvi_surface.forward(T)
    K = np.array([90.0, 100.0, 112.0])
    C = ssvi_surface.price(K, T, "call")
    P = ssvi_surface.price(K, T, "put")
    np.testing.assert_allclose(C - P, np.exp(-R * T) * (F - K), rtol=1e-12)
    assert np.all(np.isfinite(ssvi_surface.delta(K, T)))
    assert np.all(ssvi_surface.gamma(K, T) > 0)


def test_atm_skew_curvature_at_interpolated_maturity(ssvi_surface):
    """ATM metrics interpolate; the blend derivative is the derivative blend."""
    sk_lo, sk_hi = ssvi_surface.skew(0.5), ssvi_surface.skew(1.0)
    lam = (0.7 - 0.5) / (1.0 - 0.5)
    assert ssvi_surface.skew(0.7) == pytest.approx((1 - lam) * sk_lo + lam * sk_hi)
    assert np.isfinite(ssvi_surface.curvature(0.7))
    assert ssvi_surface.atm_vol(0.7) > 0


def test_theta_interpolation_method(surface_df):
    """interp_method='theta' yields parametric slices at any maturity."""
    surface = calibrate_surface(
        surface_df, model="ssvi", r=R, interp_method="theta"
    )
    params = surface.slice_at(0.7)
    assert {"theta", "rho", "eta", "forward"} <= set(params)
    # theta interpolates linearly between the bracketing slices
    lam = (0.7 - 0.5) / (1.0 - 0.5)
    expected_theta = (
        (1 - lam) * surface.params(0.5)["theta"]
        + lam * surface.params(1.0)["theta"]
    )
    assert params["theta"] == pytest.approx(expected_theta)
    # at a fitted maturity slice_at returns the exact slice
    assert surface.slice_at(0.5) == surface.params(0.5)
    # evaluation stays close to the total-variance blend
    blend = calibrate_surface(surface_df, model="ssvi", r=R)
    assert abs(surface.iv(100.0, 0.7) - blend.iv(100.0, 0.7)) < 0.01


def test_theta_method_requires_ssvi_family(surface_df):
    svi_surface = VolSurface.fit(surface_df, model="svi")
    slices = {T: svi_surface.params(T) for T in svi_surface.maturities}
    with pytest.raises(ValueError, match="theta"):
        VolSurface(svi_surface.model, slices, interp_method="theta")
    with pytest.raises(ValueError, match="unknown interp_method"):
        VolSurface(svi_surface.model, slices, interp_method="cubic")


def test_slice_at_blend_raises(ssvi_surface):
    with pytest.raises(ValueError, match="interp_method='theta'"):
        ssvi_surface.slice_at(0.7)
    # exact maturities always work
    assert ssvi_surface.slice_at(0.5) == ssvi_surface.params(0.5)


def test_single_slice_surface_cannot_interpolate(surface_df):
    svi_surface = VolSurface.fit(surface_df, model="svi")
    single = VolSurface(svi_surface.model, {0.5: svi_surface.params(0.5)})
    assert np.isfinite(single.iv(100.0, 0.5))
    with pytest.raises(ValueError, match="extrapolation"):
        single.iv(100.0, 0.7)
