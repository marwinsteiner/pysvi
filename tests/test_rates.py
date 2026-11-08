"""Rate views: flat, DiscountCurve, interest-rate model, custom callable.

interest-rate-models is a core dependency; every rate input resolves
through calibration._rate_at, so users can express a view of rates as
a flat float, an irm DiscountCurve, an irm model (its implied zero
curve from today), or any callable such as a cubic spline over pillars.
"""

import interest_rate_models as irm
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from src.pysvi import OptionChain
from src.pysvi.calibration import _rate_at

from tests.test_chain import _raw_chain, R

PILLARS_T = np.array([0.25, 5.0, 10.0, 30.0])
PILLARS_R = np.array([0.0389, 0.0469, 0.0483, 0.0517])


def test_rate_at_accepts_all_forms():
    curve = irm.DiscountCurve.from_zero_rates(PILLARS_T, PILLARS_R)
    hw = irm.get_model("hull-white", curve=curve, a=0.1, sigma=0.01)
    spline = CubicSpline(PILLARS_T, PILLARS_R)
    tte = np.array([0.5, 2.0, 7.0])

    flat = _rate_at(0.04, tte)
    assert flat == 0.04

    r_curve = _rate_at(curve, tte)
    assert r_curve.shape == tte.shape and np.all(np.isfinite(r_curve))
    # exact at a pillar
    assert float(_rate_at(curve, 5.0)) == pytest.approx(0.0469, abs=1e-12)

    # a no-arbitrage model fitted to the curve reproduces its zero rates
    r_hw = _rate_at(hw, tte)
    np.testing.assert_allclose(r_hw, r_curve, atol=1e-10)

    # arbitrary callables (cubic spline) work and hit the pillars
    r_sp = _rate_at(spline, tte)
    assert np.all(np.isfinite(r_sp))
    assert float(_rate_at(spline, 10.0)) == pytest.approx(0.0483, abs=1e-12)


def test_rate_at_scalar_and_shape():
    curve = irm.DiscountCurve.from_zero_rates(PILLARS_T, PILLARS_R)
    scalar = _rate_at(curve, 5.0)
    assert float(scalar) == pytest.approx(0.0469, abs=1e-12)
    grid = np.linspace(0.5, 20.0, 7).reshape(7, 1)
    assert _rate_at(curve, grid).shape == (7, 1)


def test_chain_with_flat_discount_curve_matches_flat_float():
    """A flat DiscountCurve reproduces the flat-float forwards exactly
    at pillar maturities and to interpolation accuracy elsewhere."""
    raw = _raw_chain()
    chain_float = OptionChain.from_dataframe(raw, rate=R)
    flat_curve = irm.DiscountCurve.flat(R)
    chain_curve = OptionChain.from_dataframe(raw, rate=flat_curve)
    for T in chain_float.maturities:
        f1 = float(chain_float.panel.query("maturity == @T")["implied_forward"].iloc[0])
        f2 = float(chain_curve.panel.query("maturity == @T")["implied_forward"].iloc[0])
        assert f2 == pytest.approx(f1, rel=1e-12)


def test_chain_with_model_and_spline_rate():
    """An irm model and a cubic spline both drive the full ingestion."""
    raw = _raw_chain()
    curve = irm.DiscountCurve.flat(R)
    hw = irm.get_model("hull-white", curve=curve, a=0.1, sigma=0.01)
    chain_hw = OptionChain.from_dataframe(raw, rate=hw)
    spline = CubicSpline(PILLARS_T, np.full(4, R))
    chain_sp = OptionChain.from_dataframe(raw, rate=spline)
    for chain in (chain_hw, chain_sp):
        for T in chain.maturities:
            F = float(chain.panel.query("maturity == @T")["implied_forward"].iloc[0])
            assert F == pytest.approx(100.0 * np.exp(R * T), rel=1e-6)
