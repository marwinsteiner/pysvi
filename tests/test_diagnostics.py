"""Arbitrage diagnostics: butterfly, Lee bounds, calendar, reports."""

import numpy as np
import pytest

from src.pysvi.models import SVI, SABR, ArbitrageFreedom
from src.pysvi.calibration import calibrate_slice
from src.pysvi.diagnostics import (
    check_slice_arbitrage, check_arbitrage, LEE_BOUND,
)

GOOD_SVI = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}

# Axel Vogt parameters [Gatheral & Jacquier 2014, section 3]: positive total
# variance yet butterfly-arbitrageable (negative density around k ~ 0.4-1.0).
VOGT_SVI = {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153}

# Right wing slope b(1 + rho) = 2.25 > 2: violates the Lee moment bound.
LEE_VIOLATING_SVI = {"a": 0.04, "b": 1.5, "rho": 0.5, "m": 0.0, "sigma": 0.1}


def test_clean_slice_passes():
    report = check_slice_arbitrage(SVI(), GOOD_SVI, maturity=0.25)
    assert report.ok
    assert report.butterfly_free and report.lee_free
    assert report.min_density > 0
    assert report.max_lee_violation == 0.0
    assert report.maturity == 0.25


def test_vogt_butterfly_violation_detected():
    """The classic arbitrageable SVI example is flagged, with location."""
    report = check_slice_arbitrage(SVI(), VOGT_SVI)
    assert not report.butterfly_free
    assert not report.ok
    assert report.min_density < -1e-3
    # violation region documented around k ~ 0.4-1.0
    assert 0.3 < report.min_density_k < 1.2


def test_lee_bound_violation_detected():
    report = check_slice_arbitrage(SVI(), LEE_VIOLATING_SVI)
    assert not report.lee_free
    assert report.right_wing_slope > LEE_BOUND
    assert report.left_wing_slope < LEE_BOUND  # b(1 - rho) = 0.75
    np.testing.assert_allclose(report.max_lee_violation, 0.25, atol=0.01)


def test_calendar_violation_detected_with_location():
    """Later slice with lower total variance is flagged, with the pair."""
    p_early = dict(GOOD_SVI)
    p_late = dict(GOOD_SVI, a=0.005)  # w shifted down: calendar arbitrage
    report = check_arbitrage(SVI(), [(0.25, p_early), (0.5, p_late)])
    assert not report.calendar_free
    assert not report.ok
    np.testing.assert_allclose(report.min_calendar_margin, -0.005, atol=1e-6)
    assert report.min_calendar_pair == (0.25, 0.5)


def test_calendar_clean_pair_passes():
    p_early = dict(GOOD_SVI)
    p_late = dict(GOOD_SVI, a=0.02)
    report = check_arbitrage(SVI(), [(0.25, p_early), (0.5, p_late)])
    assert report.calendar_free
    assert report.ok


def test_unsorted_slices_are_ordered_by_maturity():
    p_early = dict(GOOD_SVI)
    p_late = dict(GOOD_SVI, a=0.02)
    report = check_arbitrage(SVI(), [(0.5, p_late), (0.25, p_early)])
    assert report.calendar_free
    assert [s.maturity for s in report.slices] == [0.25, 0.5]


def test_single_slice_calendar_trivially_free():
    report = check_arbitrage(SVI(), [(0.25, GOOD_SVI)])
    assert report.calendar_free
    assert report.min_calendar_pair is None
    assert "single slice" in str(report)


def test_empty_slices_raise():
    with pytest.raises(ValueError, match="at least one"):
        check_arbitrage(SVI(), [])


def test_report_str_contents():
    report = check_arbitrage(
        SVI(), [(0.25, GOOD_SVI), (0.5, dict(GOOD_SVI, a=0.005))]
    )
    text = str(report)
    assert "Butterfly arbitrage: none" in text
    assert "Calendar arbitrage:    VIOLATION" in text
    assert "ARBITRAGE DETECTED" in text

    clean = check_slice_arbitrage(SVI(), GOOD_SVI)
    assert "Lee wing bounds:     satisfied" in str(clean)


def test_no_butterfly_calibration_passes_diagnostics(atm_slice):
    """End to end: NO_BUTTERFLY fit passes the butterfly check on data range."""
    model = SVI(arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY)
    params = calibrate_slice(atm_slice, model)
    assert params is not None
    report = check_slice_arbitrage(model, params, k_min=-0.7, k_max=0.7)
    assert report.butterfly_free, str(report)


def test_sabr_slice_diagnostics(atm_slice):
    """Diagnostics run on SABR via the finite-difference density."""
    model = SABR()
    T = float(atm_slice["maturity"].iloc[0])
    F = float(atm_slice["implied_forward"].iloc[0])
    params = calibrate_slice(atm_slice, model, T=T, F=F, beta=1.0)
    assert params is not None
    report = check_slice_arbitrage(model, params, maturity=T,
                                   k_min=-0.5, k_max=0.5)
    assert np.isfinite(report.min_density)
    assert report.butterfly_free, str(report)


def test_diagnostics_backend_parity(backend_mode):
    """Reports agree across numba/NumPy backends."""
    report = check_slice_arbitrage(SVI(), GOOD_SVI)
    np.testing.assert_allclose(report.min_density, 0.2452, atol=1e-3)
    assert report.ok
