"""Smile-aware Greeks (#35) and Dupire local volatility (#36)."""

import numpy as np
import pandas as pd
import pytest
from py_vollib.black import black

from src.pysvi import VarianceEvent, VolSurface, svi_total_variance
from src.pysvi.models import SVI

R = 0.02
BASE = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}


def _panel(maturities=(0.25, 0.5, 1.0, 2.0)):
    rows = []
    for T in maturities:
        F = 100.0 * np.exp(R * T)
        k = np.linspace(-0.3, 0.3, 21)
        iv = np.sqrt(svi_total_variance(k, **BASE) * (T / 0.25) / T)
        rows.append(pd.DataFrame({"strike": F * np.exp(k), "iv": iv,
                                  "maturity": T, "implied_forward": F}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def skew_surface():
    return VolSurface.fit(_panel(), model="svi", r=R,
                          initialization="multi_start")


def _flat_surface(w_of_T, maturities, interp="monotone_cubic"):
    """Direct construction of a strike-flat surface: SVI with b = 0."""
    slices = {
        T: {"a": float(w_of_T(T)), "b": 0.0, "rho": 0.0, "m": 0.0,
            "sigma": 0.1, "forward": 100.0}
        for T in maturities
    }
    return VolSurface(SVI(), slices, interp_method=interp)


# ── Sticky Greeks (#35) ──────────────────────────────────────────────

def _sticky_moneyness_price(surface, K, T, Fp, cp):
    """Black-76 price with the smile floating in log-moneyness."""
    w = surface.model.total_variance(np.log(K / Fp), surface.params(T))
    return np.array([
        black(cp, Fp, Ki, T, R, float(np.sqrt(wi / T)))
        for Ki, wi in zip(K, w)
    ])


@pytest.mark.parametrize("cp", ["c", "p"])
def test_sticky_delta_greeks_match_fd(skew_surface, cp):
    """Acceptance (#35): sticky-delta delta and gamma match finite
    differences of the price under the corresponding smile shift (bump
    F, move the smile in k)."""
    T = 0.5
    F = skew_surface.forward(T)
    K = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    h = 1e-3 * F
    up = _sticky_moneyness_price(skew_surface, K, T, F + h, cp)
    mid = _sticky_moneyness_price(skew_surface, K, T, F, cp)
    dn = _sticky_moneyness_price(skew_surface, K, T, F - h, cp)
    np.testing.assert_allclose(
        skew_surface.delta(K, T, cp=cp, sticky="delta"),
        (up - dn) / (2 * h), atol=1e-5)
    np.testing.assert_allclose(
        skew_surface.gamma(K, T, sticky="delta"),
        (up - 2 * mid + dn) / h ** 2, rtol=1e-3, atol=1e-6)


def test_sticky_strike_default_unchanged(skew_surface):
    """sticky='strike' is the default and bitwise identical to it."""
    K = np.array([90.0, 100.0, 110.0])
    for T in (0.5, 0.7):
        np.testing.assert_array_equal(
            skew_surface.delta(K, T), skew_surface.delta(K, T, sticky="strike"))
        np.testing.assert_array_equal(
            skew_surface.gamma(K, T), skew_surface.gamma(K, T, sticky="strike"))
        np.testing.assert_array_equal(
            skew_surface.theta(K, T), skew_surface.theta(K, T, sticky="strike"))


def test_derman_delta_ordering(skew_surface):
    """Acceptance (#35): with a negative skew the sticky-delta call delta
    exceeds the sticky-strike delta (Derman 1999 regime ordering)."""
    K = np.linspace(85.0, 115.0, 7)
    T = 0.5
    assert np.all(skew_surface.delta(K, T, sticky="delta")
                  > skew_surface.delta(K, T, sticky="strike"))


def test_conventions_coincide_on_flat_smile():
    """No skew, no difference: the smile-slope terms vanish."""
    s = _flat_surface(lambda T: 0.04 * T, (0.25, 0.5, 1.0), interp="total_variance")
    K = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(s.delta(K, 0.5, sticky="delta"),
                               s.delta(K, 0.5), atol=1e-12)
    np.testing.assert_allclose(s.gamma(K, 0.5, sticky="delta"),
                               s.gamma(K, 0.5), rtol=1e-10)


def test_theta_identical_across_conventions(skew_surface):
    """Theta holds F fixed, so strike and moneyness coincide."""
    K = np.array([90.0, 100.0, 110.0])
    np.testing.assert_array_equal(skew_surface.theta(K, 0.5, sticky="delta"),
                                  skew_surface.theta(K, 0.5))


def test_sticky_validation(skew_surface):
    with pytest.raises(ValueError, match="unknown sticky"):
        skew_surface.delta(100.0, 0.5, sticky="moneyness")
    with pytest.raises(NotImplementedError, match="local_vol"):
        skew_surface.gamma(100.0, 0.5, sticky="local_vol")


# ── Dupire local volatility (#36) ────────────────────────────────────

def test_flat_surface_local_vol_equals_implied():
    """Acceptance (#36): on a flat surface sigma_loc == sigma_implied."""
    s = _flat_surface(lambda T: 0.04 * T, (0.25, 0.5, 1.0, 2.0))
    K = np.array([80.0, 100.0, 125.0])
    for T in (0.25, 0.6, 1.3):
        np.testing.assert_allclose(s.local_vol(K, T), 0.2, rtol=1e-10)
        np.testing.assert_allclose(s.local_vol(K, T), s.iv(K, T), rtol=1e-10)


def test_time_dependent_generator_recovered():
    """Acceptance (#36): w(k,T) = integral of sigma^2(t) dt with
    sigma^2(t) = 0.04 + 0.02 t -- local variance recovers sigma^2(T)."""
    Ts = np.linspace(0.25, 3.0, 12)
    s = _flat_surface(lambda T: 0.04 * T + 0.01 * T * T, Ts)
    for T in (0.6, 1.1, 2.4):
        expected = np.sqrt(0.04 + 0.02 * T)
        assert float(s.local_vol(100.0, T)) == pytest.approx(expected, rel=5e-3)


def test_butterfly_violation_returns_nan_with_count():
    """Acceptance (#36): where g(k) <= 0 local variance is NaN and
    counted -- never negative or imaginary."""
    BAD = {"a": -0.006, "b": 0.4, "rho": -0.9, "m": 0.0, "sigma": 0.05}
    slices = {
        T: {k: v * (T / 0.25) if k in ("a", "b") else v for k, v in BAD.items()}
        | {"forward": 100.0}
        for T in (0.25, 0.5, 1.0)
    }
    s = VolSurface(SVI(), slices, interp_method="monotone_cubic")
    k = np.linspace(-1.5, 1.5, 301)
    lv, n_bad = s.local_variance(k, 0.6, return_n_invalid=True)
    assert n_bad > 0
    finite = lv[np.isfinite(lv)]
    assert finite.size > 0 and np.all(finite >= 0.0)


def test_calendar_violation_returns_nan():
    """Decreasing total variance in T (calendar arbitrage) gives NaN."""
    s = _flat_surface(lambda T: 0.05 - 0.01 * T, (0.25, 0.5, 1.0))
    lv, n_bad = s.local_variance(np.array([0.0, 0.1]), 0.7,
                                 return_n_invalid=True)
    assert np.all(np.isnan(lv)) and n_bad == 2


def test_local_vol_requires_c1(skew_surface):
    with pytest.raises(ValueError, match="C1"):
        skew_surface.local_vol(100.0, 0.7)


def test_local_vol_on_fitted_c1_surface():
    s = VolSurface.fit(_panel(), model="svi", r=R,
                       interp_method="monotone_cubic",
                       initialization="multi_start")
    K = np.array([90.0, 100.0, 110.0])
    lv, n_bad = s.local_vol(K, 0.7, return_n_invalid=True)
    assert n_bad == 0 and np.all(lv > 0)
    assert isinstance(s.local_vol(100.0, 0.7), float)


def test_local_vol_with_event_uses_continuous_rate():
    """An event is a jump in w, not a rate: the numerator stays the
    continuous component's dw/dT."""
    s = _flat_surface(lambda T: 0.04 * T, (0.25, 0.5, 1.0, 2.0))
    s_ev = VolSurface(s.model, {T: s.params(T) for T in s.maturities},
                      interp_method="monotone_cubic",
                      events=[VarianceEvent(0.75, 0.01)])
    lv_before = float(s_ev.local_variance(0.0, 0.6))
    lv_after = float(s_ev.local_variance(0.0, 0.9))
    assert lv_before == pytest.approx(0.04, rel=1e-8)
    assert lv_after == pytest.approx(0.04, rel=1e-8)
