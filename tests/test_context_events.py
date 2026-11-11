"""Variance events (#29), MarketContext (#30), economic arbitrage and
evaluation status (#31)."""

import numpy as np
import pandas as pd
import pytest

from src.pysvi import (
    CLASS_EXECUTABLE, CLASS_EXTRAPOLATION, CLASS_MATHEMATICAL,
    MarketContext, OptionChain, VarianceEvent, VolSurface,
    classify_arbitrage, implied_event_variances, svi_total_variance,
)

BASE = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}
R = 0.02
EV = VarianceEvent(0.4, 0.006, "earnings")


def _event_panel(maturities=(0.25, 0.75), event=EV, band=0.0):
    k = np.linspace(-0.25, 0.25, 21)
    rows = []
    for T in maturities:
        F = 100.0 * np.exp(R * T)
        w = svi_total_variance(k, **BASE) * (T / 0.25)
        if event is not None and event.time <= T:
            w = w + event.variance
        iv = np.sqrt(w / T)
        d = {"strike": F * np.exp(k), "iv": iv, "maturity": T,
             "implied_forward": F}
        if band:
            d["iv_bid"] = iv - band
            d["iv_ask"] = iv + band
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


# ── Variance events ──────────────────────────────────────────────────

def test_event_aware_interpolation_reproduces_jump():
    """Acceptance (#29): interpolation reproduces the discrete jump at
    the event time; the naive blend smears it (documented wrong)."""
    df = _event_panel()
    s_ev = VolSurface.fit(df, model="svi", events=[EV],
                          initialization="multi_start")
    s_naive = VolSurface.fit(df, model="svi", initialization="multi_start")
    jump = (s_ev.total_variance(0.0, EV.time + 1e-4)
            - s_ev.total_variance(0.0, EV.time - 1e-4))
    assert jump == pytest.approx(EV.variance, rel=0.05)
    naive_jump = (s_naive.total_variance(0.0, EV.time + 1e-4)
                  - s_naive.total_variance(0.0, EV.time - 1e-4))
    assert naive_jump < 0.2 * EV.variance  # smeared away
    # truth on both sides of the event
    for T in (0.39, 0.41):
        w_true = (float(svi_total_variance(np.array([0.0]), **BASE)[0])
                  * (T / 0.25) + (EV.variance if T >= EV.time else 0.0))
        assert s_ev.total_variance(0.0, T) == pytest.approx(w_true, rel=1e-3)


def test_calendar_diagnostics_clean_across_event():
    """Acceptance (#29): the calendar check sees the continuous
    component, so a genuine event raises no false violation."""
    df = _event_panel()
    s = VolSurface.fit(df, model="svi", events=[EV],
                       initialization="multi_start")
    assert s.check_arbitrage().calendar_free
    assert s.diagnose().ok


def test_events_serialize_and_validate(tmp_path):
    df = _event_panel()
    s = VolSurface.fit(df, model="svi", events=[EV],
                       initialization="multi_start")
    path = tmp_path / "ev.json"
    s.save(path)
    loaded = VolSurface.load(path)
    assert loaded.events == s.events
    assert loaded.total_variance(0.0, 0.5) == s.total_variance(0.0, 0.5)
    with pytest.raises(ValueError, match="variance must be non-negative"):
        VarianceEvent(0.4, -0.01)
    with pytest.raises(ValueError, match="time must be positive"):
        VarianceEvent(0.0, 0.01)


def test_oversized_event_rejected():
    """An event variance exceeding the quoted total variance is
    inconsistent with the market: strict raises, warn skips the slice."""
    big = VarianceEvent(0.4, 10.0)
    df = _event_panel(event=None)  # market does NOT contain the event
    with pytest.raises(ValueError, match="event variance"):
        VolSurface.fit(df, model="svi", events=[big], mode="strict")


def test_implied_event_variances():
    df = _event_panel()
    (rep,) = implied_event_variances(df, [EV])
    assert rep["T_before"] == 0.25 and rep["T_after"] == 0.75
    # quoted jump upper-bounds the event variance (contains cont growth)
    assert rep["quoted_jump"] > EV.variance
    assert rep["specified"] == EV.variance


# ── MarketContext ────────────────────────────────────────────────────

def _dated_chain(day_count="ACT/365F"):
    ctx = MarketContext("2026-09-24", spot=100.0, rate=R,
                        day_count=day_count)
    from py_vollib.black import black
    k = np.linspace(-0.2, 0.2, 15)
    rows = []
    for expiry in ("2026-12-24", "2027-03-24"):
        T = ctx.year_fraction(expiry)
        F = 100.0 * np.exp(R * T)
        iv = np.sqrt(svi_total_variance(k, **BASE) * (T / 0.25) / T)
        for ki, vi in zip(k, iv):
            K = F * np.exp(ki)
            for flag in ("c", "p"):
                px = black(flag, F, K, T, R, float(vi))
                rows.append({"strike": K, "expiry": expiry, "cp": flag,
                             "bid": max(px - 0.01, 1e-4), "ask": px + 0.01})
    return ctx, pd.DataFrame(rows)


def test_context_resolves_dates_and_prices_consistently():
    """Acceptance (#30): a chain built from a MarketContext prices
    consistently -- put-call parity holds against the context's own
    discount factors."""
    ctx, raw = _dated_chain()
    chain = OptionChain.from_dataframe(raw, context=ctx)
    surface = chain.fit(model="svi", initialization="multi_start")
    T = float(chain.maturities[0])
    K = np.array([95.0, 100.0, 105.0])
    C = np.atleast_1d(surface.price(K, T, cp="call"))
    P = np.atleast_1d(surface.price(K, T, cp="put"))
    D = ctx.discount(T)
    F = surface.forward(T)
    np.testing.assert_allclose(C - P, D * (F - K), rtol=1e-9)


def test_mixed_conventions_loudly_rejected():
    """Acceptance (#30): context plus separate numeraire inputs raise."""
    ctx, raw = _dated_chain()
    with pytest.raises(ValueError, match="EITHER context"):
        OptionChain.from_dataframe(raw, context=ctx, rate=0.05)
    with pytest.raises(ValueError, match="EITHER context"):
        OptionChain.from_dataframe(raw, context=ctx, spot=100.0)


def test_day_count_changes_T_and_iv():
    """Acceptance (#30): the day-count choice demonstrably changes T
    and therefore the inverted implied vols."""
    ctx365, raw = _dated_chain("ACT/365F")
    ctx360, _ = _dated_chain("ACT/360")
    T365 = ctx365.year_fraction("2026-12-24")
    T360 = ctx360.year_fraction("2026-12-24")
    assert T360 > T365  # same days, smaller denominator
    c365 = OptionChain.from_dataframe(raw, context=ctx365)
    c360 = OptionChain.from_dataframe(raw, context=ctx360)
    iv365 = c365.panel["iv"].median()
    iv360 = c360.panel["iv"].median()
    assert abs(iv365 - iv360) / iv365 > 0.005  # same prices, different T
    assert MarketContext("2026-09-24", day_count="BUS/252").year_fraction(
        "2026-09-28") == pytest.approx(2.0 / 252.0)  # Thu->Mon = 2 bus days
    with pytest.raises(ValueError, match="unknown day_count"):
        MarketContext("2026-09-24", day_count="ACT/364")


# ── Evaluation status and economic classification ────────────────────

def test_iv_return_status_labels(surface_df):
    from tests.conftest import SURFACE_RATE as SR
    s = VolSurface.fit(surface_df, model="svi", r=SR,
                       initialization="multi_start")
    T_fit = float(s.maturities[0])
    F = s.forward(T_fit)
    K = np.array([F * np.exp(-0.6), F, F * np.exp(0.6)])
    iv, status = s.iv(K, T_fit, return_status=True)
    assert list(status) == ["extrapolated", "observed", "extrapolated"]
    T_mid = 0.5 * (float(s.maturities[0]) + float(s.maturities[1]))
    _, status_mid = s.iv(np.array([s.forward(T_mid)]), T_mid,
                         return_status=True)
    assert status_mid[0] == "interpolated"
    iv_s, st_s = s.iv(float(F), T_fit, return_status=True)
    assert isinstance(iv_s, float) and st_s == "observed"
    bare = VolSurface(s.model, {T: s.params(T) for T in s.maturities})
    with pytest.raises(ValueError, match="fit report"):
        bare.iv(K, T_fit, return_status=True)


BAD = {"a": -0.006, "b": 0.4, "rho": -0.9, "m": 0.0, "sigma": 0.05}


def _bad_panel(band=0.02, crossed=False):
    k = np.linspace(-0.15, 0.15, 21)
    rows = []
    for T in (0.25, 0.5):
        F = 100.0 * np.exp(R * T)
        w = svi_total_variance(k, **BAD) * (T / 0.25)
        iv = np.sqrt(w / T)
        lo, hi = iv - band, iv + band
        if crossed:
            lo, hi = iv + band, iv - band  # crossed quotes
        rows.append(pd.DataFrame({
            "strike": F * np.exp(k), "iv": iv, "maturity": T,
            "implied_forward": F, "iv_bid": lo, "iv_ask": hi,
        }))
    return pd.concat(rows, ignore_index=True)


def test_classification_quote_consistent_vs_executable():
    """Acceptance (#31): wide quotes make the violation
    quote-consistent; crossed quotes make it executable."""
    panel = _bad_panel(band=0.05)
    s = VolSurface.fit(panel, model="svi", initialization="multi_start")
    findings = classify_arbitrage(s, panel=panel)
    assert any(f.classification == "quote_consistent" for f in findings
               if f.finding != "none")
    crossed = _bad_panel(band=0.02, crossed=True)
    s2 = VolSurface.fit(crossed, model="svi", initialization="multi_start")
    findings2 = classify_arbitrage(s2, panel=crossed)
    assert any(f.classification == CLASS_EXECUTABLE for f in findings2)


def test_classification_without_panel_is_mathematical():
    panel = _bad_panel()
    s = VolSurface.fit(panel, model="svi", initialization="multi_start")
    findings = classify_arbitrage(s)
    assert all(f.classification == CLASS_MATHEMATICAL for f in findings
               if f.finding != "none")
    assert any(f.finding != "none" for f in findings)


def test_wing_violation_outside_range_is_extrapolation_risk():
    """Acceptance (#31): a violation beyond the quoted strikes is
    extrapolation risk, not an executable arbitrage."""
    # narrow quoted range around ATM; wings of the fit violate outside
    k = np.linspace(-0.04, 0.04, 15)
    rows = []
    for T in (0.25, 0.5):
        F = 100.0 * np.exp(R * T)
        w = svi_total_variance(k, **BAD) * (T / 0.25)
        rows.append(pd.DataFrame({
            "strike": F * np.exp(k), "iv": np.sqrt(w / T), "maturity": T,
            "implied_forward": F,
        }))
    panel = pd.concat(rows, ignore_index=True)
    s = VolSurface.fit(panel, model="svi", initialization="multi_start")
    report = s.check_arbitrage()  # wide default grid: wings violate
    if not report.ok:
        findings = classify_arbitrage(s, panel=panel)
        bad = [f for f in findings if f.finding != "none"]
        assert bad and all(
            f.classification in (CLASS_EXTRAPOLATION, CLASS_MATHEMATICAL,
                                 "quote_consistent")
            for f in bad
        )
        assert any(f.classification == CLASS_EXTRAPOLATION for f in bad)
