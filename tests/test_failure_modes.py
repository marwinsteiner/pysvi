"""Strict/warn/lenient failure semantics (issue #24)."""

import numpy as np
import pandas as pd
import pytest

from py_vollib.black import black

from src.pysvi import OptionChain, VolSurface, calibrate_surface, svi_total_variance

from tests.test_chain import _raw_chain, R


def test_mode_validated_everywhere(raw_chain_df):
    with pytest.raises(ValueError, match="unknown mode"):
        OptionChain.from_dataframe(raw_chain_df, rate=R, mode="silent")
    chain = OptionChain.from_dataframe(raw_chain_df, rate=R)
    with pytest.raises(ValueError, match="unknown mode"):
        chain.fit(mode="permissive")
    with pytest.raises(ValueError, match="unknown mode"):
        VolSurface.fit(chain.panel, mode="loose")
    with pytest.raises(ValueError, match="unknown mode"):
        calibrate_surface(chain.panel, mode="hard")


@pytest.fixture
def raw_chain_df() -> pd.DataFrame:
    return _raw_chain()


def _with_crossed_quote(raw):
    return pd.concat([raw, pd.DataFrame([{
        "strike": 100.0, "expiry": 0.25, "cp": "c", "bid": 5.0, "ask": 4.0,
    }])], ignore_index=True)


def test_strict_raises_on_bad_quote_with_location(raw_chain_df):
    dirty = _with_crossed_quote(raw_chain_df)
    with pytest.raises(ValueError, match=r"strict.*1 invalid quote"):
        OptionChain.from_dataframe(dirty, rate=R, mode="strict")
    # the offending input row index is named
    with pytest.raises(ValueError, match=str(len(raw_chain_df))):
        OptionChain.from_dataframe(dirty, rate=R, mode="strict")


def test_strict_raises_on_unformable_expiry():
    calls_only = _raw_chain(maturities=(0.5,))
    calls_only = calls_only[calls_only["cp"] == "c"]
    with pytest.raises(ValueError, match=r"strict.*no put-call pairs"):
        OptionChain.from_dataframe(calls_only, rate=R, mode="strict")


def test_lenient_is_silent_but_recorded(raw_chain_df, capsys):
    from loguru import logger
    import io
    sink = io.StringIO()
    token = logger.add(sink, level="WARNING")
    try:
        dirty = _with_crossed_quote(raw_chain_df)
        chain = OptionChain.from_dataframe(dirty, rate=R, mode="lenient")
    finally:
        logger.remove(token)
    assert "invalid quote" not in sink.getvalue()      # silent
    assert chain.rejections["rejected_quotes"] == 1    # but recorded


def test_rejections_recorded_and_carried_to_report(raw_chain_df):
    dirty = _with_crossed_quote(raw_chain_df)
    chain = OptionChain.from_dataframe(dirty, rate=R)
    assert chain.rejections["rejected_quotes"] == 1
    assert chain.rejections["failed_inversions"] == 0
    assert chain.rejections["skipped_expiries"] == []
    surface = chain.fit(model="svi", initialization="multi_start")
    assert surface.fit_report.n_rejected_quotes == 1
    assert surface.fit_report.n_failed_inversions == 0
    assert surface.fit_report.n_skipped_expiries == 0
    assert "1 quotes rejected" in surface.fit_report.summary()


def test_clean_chain_report_has_no_ingestion_line(raw_chain_df):
    surface = OptionChain.from_dataframe(raw_chain_df, rate=R).fit(
        model="svi", initialization="multi_start")
    assert surface.fit_report.n_rejected_quotes == 0
    assert "Ingestion" not in surface.fit_report.summary()


def _panel_with_thin_slice():
    rows = []
    for T, n in ((0.25, 21), (0.5, 3)):  # second slice below min_points
        F = 100.0 * np.exp(R * T)
        k = np.linspace(-0.2, 0.2, n)
        iv = np.sqrt(svi_total_variance(
            k, 0.01, 0.12, -0.6, 0.01, 0.25) * (T / 0.25) / T)
        for ki, vi in zip(k, iv):
            rows.append({"strike": F * np.exp(ki), "iv": vi,
                         "maturity": T, "implied_forward": F})
    return pd.DataFrame(rows)


def test_fit_strict_raises_on_thin_slice():
    panel = _panel_with_thin_slice()
    with pytest.raises(ValueError, match=r"strict.*T=0\.5.*insufficient"):
        VolSurface.fit(panel, model="svi", mode="strict")
    with pytest.raises(ValueError, match=r"strict.*T=0\.5.*insufficient"):
        calibrate_surface(panel, model="svi", enforce_calendar=False,
                          mode="strict")


def test_fit_lenient_skips_silently_but_reports():
    from loguru import logger
    import io
    sink = io.StringIO()
    token = logger.add(sink, level="WARNING")
    try:
        surface = VolSurface.fit(_panel_with_thin_slice(), model="svi",
                                 mode="lenient", initialization="multi_start")
    finally:
        logger.remove(token)
    assert "insufficient" not in sink.getvalue()
    assert surface.fit_report.n_failed == 1  # still on the record


def test_serialization_round_trips_ingestion_counts(raw_chain_df, tmp_path):
    dirty = _with_crossed_quote(raw_chain_df)
    surface = OptionChain.from_dataframe(dirty, rate=R).fit(
        model="svi", initialization="multi_start")
    path = tmp_path / "s.json"
    surface.save(path)
    loaded = VolSurface.load(path)
    assert loaded.fit_report.n_rejected_quotes == 1
    assert str(loaded.fit_report) == str(surface.fit_report)


def test_no_blanket_warning_suppression():
    """Importing pysvi must not install a catch-all warnings filter."""
    import warnings
    import pysvi  # noqa: F401  (the installed package, freshly resolved)
    for action, message, category, module, lineno in warnings.filters:
        assert not (
            action == "ignore" and message is None
            and category is Warning and module is None
        ), "blanket warnings.filterwarnings('ignore') found"
