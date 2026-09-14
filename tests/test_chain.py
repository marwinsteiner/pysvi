"""OptionChain ingestion: quotes to panel to surface."""

import numpy as np
import pandas as pd
import pytest

from py_vollib.black import black

from src.pysvi import OptionChain, svi_total_variance

R = 0.02
BASE = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}


def _raw_chain(maturities=(0.25, 0.5, 1.0), n=21, spread=0.02):
    rows = []
    for T in maturities:
        F = 100.0 * np.exp(R * T)
        k = np.linspace(-0.25, 0.25, n)
        iv = np.sqrt(svi_total_variance(k, **BASE) * (T / 0.25) / T)
        for ki, vi in zip(k, iv):
            K = F * np.exp(ki)
            for flag in ("c", "p"):
                px = black(flag, F, K, T, R, float(vi))
                sp = max(spread, 0.01 * px)
                rows.append({
                    "strike": K, "expiry": T, "cp": flag,
                    "bid": max(px - sp / 2, 1e-3), "ask": px + sp / 2,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def raw_chain() -> pd.DataFrame:
    return _raw_chain()


def test_panel_schema_and_forwards(raw_chain):
    chain = OptionChain.from_dataframe(raw_chain, rate=R)
    panel = chain.panel
    assert {"strike", "iv", "maturity", "implied_forward", "iv_bid", "iv_ask"} <= set(panel)
    for T in chain.maturities:
        F = float(panel[panel["maturity"] == T]["implied_forward"].iloc[0])
        assert F == pytest.approx(100.0 * np.exp(R * T), rel=1e-6)


def test_recovered_ivs_match_input(raw_chain):
    """Mid IVs recover the generating smile within half the spread."""
    chain = OptionChain.from_dataframe(raw_chain, rate=R)
    panel = chain.panel
    for T in chain.maturities:
        g = panel[panel["maturity"] == T]
        k = np.log(g["strike"].to_numpy() / g["implied_forward"].iloc[0])
        iv_true = np.sqrt(svi_total_variance(k, **BASE) * (T / 0.25) / T)
        assert float(np.nanmax(np.abs(g["iv"].to_numpy() - iv_true))) < 0.01
        # bid/ask bracket the mid
        ok = np.isfinite(g["iv_bid"]) & np.isfinite(g["iv_ask"])
        assert (g["iv_bid"][ok] <= g["iv"][ok] + 1e-12).all()
        assert (g["iv_ask"][ok] >= g["iv"][ok] - 1e-12).all()


def test_two_call_end_to_end(raw_chain):
    """Raw quotes to fitted surface in two calls, with a clean report."""
    chain = OptionChain.from_dataframe(raw_chain, rate=R)
    surface = chain.fit(model="svi", enforce_calendar=True,
                        initialization="multi_start")
    assert surface.fit_report.ok
    assert surface.diagnose().ok
    assert surface.r == pytest.approx(R)  # chain's flat rate becomes pricing r


def test_term_structure_rate(raw_chain):
    """A callable rate reproduces the flat result when constant."""
    chain = OptionChain.from_dataframe(raw_chain, rate=lambda T: R)
    for T in chain.maturities:
        F = float(chain.panel[chain.panel["maturity"] == T]["implied_forward"].iloc[0])
        assert F == pytest.approx(100.0 * np.exp(R * T), rel=1e-6)
    # callable rate has no flat surface representation; defaults to 0
    surface = chain.fit(model="svi")
    assert surface.r == 0.0


def test_forward_fallback_without_pairs():
    """Call-only expiry uses spot * exp((r - q) T)."""
    raw = _raw_chain(maturities=(0.5,))
    calls_only = raw[raw["cp"] == "c"]
    chain = OptionChain.from_dataframe(calls_only, rate=R, dividend_yield=0.005,
                                       spot=100.0)
    F = float(chain.panel["implied_forward"].iloc[0])
    assert F == pytest.approx(100.0 * np.exp((R - 0.005) * 0.5), rel=1e-9)
    # without spot the expiry is skipped -> no usable slice
    with pytest.raises(ValueError, match="no expiry produced"):
        OptionChain.from_dataframe(calls_only, rate=R)


def test_invalid_rows_dropped_and_cp_validated(raw_chain):
    dirty = pd.concat([raw_chain, pd.DataFrame([{
        "strike": 100.0, "expiry": 0.25, "cp": "c", "bid": 5.0, "ask": 4.0,  # crossed
    }])], ignore_index=True)
    chain = OptionChain.from_dataframe(dirty, rate=R)
    assert len(chain.panel) == 63  # crossed quote dropped

    bad = raw_chain.copy()
    bad.loc[0, "cp"] = "straddle"
    with pytest.raises(ValueError, match="unrecognized cp"):
        OptionChain.from_dataframe(bad, rate=R)


def test_custom_column_names(raw_chain):
    renamed = raw_chain.rename(columns={
        "strike": "K", "expiry": "tte", "cp": "flag", "bid": "b", "ask": "a",
    })
    chain = OptionChain.from_dataframe(
        renamed, strike="K", expiry="tte", cp="flag", bid="b", ask="a", rate=R,
    )
    assert len(chain.panel) == 63
