"""VolSurface: fitting, evaluation, diagnostics, Black-76 pricing."""

import numpy as np
import pytest

from src.pysvi.surface import VolSurface
from src.pysvi.calibration import get_model

from tests.conftest import SURFACE_RATE as R


def _slice_iv_rmse(surface, df, T):
    g = df[df["maturity"] == T]
    fit_iv = surface.iv(g["strike"].to_numpy(), T)
    return float(np.sqrt(np.mean((g["iv"].to_numpy() - fit_iv) ** 2)))


@pytest.fixture
def svi_surface(surface_df, backend_mode) -> VolSurface:
    return VolSurface.fit(surface_df, model="svi", r=R)


# ── Fitting ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "model", ["svi", "natural", "ssvi", "essvi", "jw", "dsvi", "sabr"]
)
def test_fit_all_models(surface_df, model):
    """VolSurface.fit works for every parametrization with one schema."""
    surface = VolSurface.fit(surface_df, model=model, r=R)
    np.testing.assert_allclose(surface.maturities, [0.25, 0.5, 1.0])
    for T in surface.maturities:
        # 0.025 matches the historical per-model roundtrip thresholds
        assert _slice_iv_rmse(surface, surface_df, T) < 0.025, (model, T)
        assert surface.params(T)["forward"] == pytest.approx(
            100.0 * np.exp(R * T)
        )


def test_fit_accepts_model_instance(surface_df):
    surface = VolSurface.fit(surface_df, model=get_model("svi"))
    assert len(surface.maturities) == 3


def test_fit_forwards_calibration_controls(surface_df):
    """Calibration-control kwargs pass through to every slice."""
    surface = VolSurface.fit(
        surface_df, model="svi", initialization="multi_start", loss="soft_l1"
    )
    for T in surface.maturities:
        assert _slice_iv_rmse(surface, surface_df, T) < 0.01


def test_fit_skips_failed_slices(surface_df):
    """A slice that cannot calibrate is skipped, not fatal."""
    import pandas as pd
    junk = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0],  # < min_points after cleaning
        "iv": [0.2, 0.21, 0.22],
        "maturity": 2.0,
        "implied_forward": 100.0,
    })
    surface = VolSurface.fit(pd.concat([surface_df, junk]), model="svi")
    np.testing.assert_allclose(surface.maturities, [0.25, 0.5, 1.0])

    with pytest.raises(ValueError, match="no slice calibrated"):
        VolSurface.fit(junk, model="svi")


# ── Slice access and evaluation ──────────────────────────────────────

def test_unfitted_maturity_raises(svi_surface):
    with pytest.raises(ValueError, match="not a fitted slice"):
        svi_surface.iv(100.0, 0.7)


def test_forward_and_params(svi_surface):
    assert svi_surface.forward(0.5) == pytest.approx(100.0 * np.exp(R * 0.5))
    params = svi_surface.params(0.5)
    assert {"a", "b", "rho", "m", "sigma", "forward"} <= set(params)
    # params() returns a copy
    params["a"] = 999.0
    assert svi_surface.params(0.5)["a"] != 999.0


def test_atm_skew_curvature_consistency(svi_surface, surface_df):
    """ATM metrics agree with the underlying model derivative API."""
    T = 0.5
    params = svi_surface.params(T)
    zero = np.array([0.0])
    assert svi_surface.atm_vol(T) == pytest.approx(
        float(np.sqrt(svi_surface.model.total_variance(zero, params)[0] / T))
    )
    assert svi_surface.skew(T) == pytest.approx(
        float(svi_surface.model.dw_dk(zero, params)[0])
    )
    assert svi_surface.curvature(T) == pytest.approx(
        float(svi_surface.model.d2w_dk2(zero, params)[0])
    )
    # ATM vol tracks the data (flat 0.404-vol fixture)
    assert abs(svi_surface.atm_vol(T) - 0.404) < 5e-3


def test_evaluation_shapes(svi_surface):
    T = 0.25
    assert isinstance(svi_surface.iv(100.0, T), float)
    assert svi_surface.iv(np.array([95.0, 100.0, 105.0]), T).shape == (3,)
    assert isinstance(svi_surface.total_variance(0.0, T), float)
    assert svi_surface.total_variance(np.linspace(-0.1, 0.1, 5), T).shape == (5,)


def test_direct_construction(svi_surface):
    """A surface rebuilt from its own slices evaluates identically."""
    rebuilt = VolSurface(
        svi_surface.model,
        {T: svi_surface.params(T) for T in svi_surface.maturities},
        r=R,
    )
    K = np.array([95.0, 100.0, 105.0])
    np.testing.assert_allclose(rebuilt.iv(K, 0.5), svi_surface.iv(K, 0.5))

    with pytest.raises(ValueError, match="forward"):
        params = svi_surface.params(0.5)
        params.pop("forward")
        VolSurface(svi_surface.model, {0.5: params})
    with pytest.raises(ValueError, match="duplicate"):
        p = svi_surface.params(0.5)
        VolSurface(svi_surface.model, [(0.5, p), (0.5, dict(p))])
    with pytest.raises(ValueError, match="at least one"):
        VolSurface(svi_surface.model, {})


# ── Diagnostics ──────────────────────────────────────────────────────

def test_check_arbitrage_on_data_range(svi_surface):
    """The fitted surface verifies clean on the quoted strike range."""
    report = svi_surface.check_arbitrage(k_data=np.linspace(-0.25, 0.25, 21))
    assert report.ok, str(report)
    assert report.calendar_free


# ── Black-76 pricing and Greeks ──────────────────────────────────────

def test_price_matches_py_vollib(svi_surface):
    from py_vollib.black import black
    T = 0.5
    F = svi_surface.forward(T)
    for K in (90.0, 100.0, 110.0):
        sigma = svi_surface.iv(K, T)
        for cp, flag in (("call", "c"), ("put", "p")):
            expected = black(flag, F, K, T, R, sigma)
            np.testing.assert_allclose(
                svi_surface.price(K, T, cp), expected, rtol=1e-10
            )


def test_put_call_parity(svi_surface):
    T = 1.0
    F = svi_surface.forward(T)
    K = np.array([80.0, 100.0, 120.0])
    C = svi_surface.price(K, T, "call")
    P = svi_surface.price(K, T, "put")
    np.testing.assert_allclose(C - P, np.exp(-R * T) * (F - K), rtol=1e-12)


def test_greeks_match_finite_differences(svi_surface):
    """delta/gamma/vega/theta agree with FD of the Black price at fixed sigma."""
    from py_vollib.black import black
    T = 0.5
    F = svi_surface.forward(T)
    h = 1e-4
    for K in (92.0, 100.0, 109.0):
        sigma = svi_surface.iv(K, T)
        d_fd = (black("c", F + h, K, T, R, sigma) - black("c", F - h, K, T, R, sigma)) / (2 * h)
        g_fd = (
            black("c", F + h, K, T, R, sigma)
            - 2 * black("c", F, K, T, R, sigma)
            + black("c", F - h, K, T, R, sigma)
        ) / h**2
        v_fd = (black("c", F, K, T, R, sigma + h) - black("c", F, K, T, R, sigma - h)) / (2 * h)
        t_fd = -(black("c", F, K, T, R, sigma) - black("c", F, K, T - 1e-5, R, sigma)) / 1e-5
        np.testing.assert_allclose(svi_surface.delta(K, T, "call"), d_fd, rtol=1e-5)
        np.testing.assert_allclose(svi_surface.gamma(K, T), g_fd, rtol=1e-3)
        np.testing.assert_allclose(svi_surface.vega(K, T), v_fd, rtol=1e-5)
        np.testing.assert_allclose(svi_surface.theta(K, T, "call"), t_fd, rtol=1e-3)


def test_put_delta_and_theta(svi_surface):
    """Put delta = call delta - e^{-rT}; put theta via parity."""
    T = 0.5
    F = svi_surface.forward(T)
    K = 100.0
    disc = np.exp(-R * T)
    np.testing.assert_allclose(
        svi_surface.delta(K, T, "put"),
        svi_surface.delta(K, T, "call") - disc,
        rtol=1e-12,
    )
    # d/dt of parity P = C - e^{-rT}(F - K): theta_P = theta_C - r e^{-rT}(F - K)
    np.testing.assert_allclose(
        svi_surface.theta(K, T, "put"),
        svi_surface.theta(K, T, "call") - R * disc * (F - K),
        rtol=1e-10,
    )


def test_pricing_shapes_and_validation(svi_surface):
    T = 0.25
    K = np.array([95.0, 100.0, 105.0])
    assert svi_surface.price(K, T).shape == (3,)
    assert isinstance(svi_surface.price(100.0, T), float)
    with pytest.raises(ValueError, match="cp"):
        svi_surface.price(100.0, T, cp="straddle")
    with pytest.raises(ValueError, match="positive"):
        svi_surface.price(-5.0, T)
