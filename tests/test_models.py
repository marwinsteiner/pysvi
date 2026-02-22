import pytest
from hypothesis import given, strategies as st
from src.pysvi.models import *
from src.pysvi.models import _butterfly_penalty, _calendar_penalty, _svi_derivatives, _ssvi_derivatives
from src.pysvi.calibration import get_model


def test_svi_total_variance_formula():
    """Verify implementation matches published formula."""
    k = np.array([0.0, 0.1, -0.1])
    # a + b{ρ(k-m) + sqrt[(k-m)² + σ²]}
    expected = np.array([0.03, 0.0274, 0.0374])
    result = svi_total_variance(k, a=0.01, b=0.1, rho=-0.5, m=0.0, sigma=0.2)
    np.testing.assert_allclose(result, expected, rtol=1e-2)


@given(k=st.lists(st.floats(-1.0, 1.0), min_size=1, max_size=100))
def test_svi_positivity(k):
    """SVI total variance always non-negative."""
    k = np.array(k)
    w = svi_total_variance(k, 0.01, 0.1, -0.5, 0.0, 0.2)
    assert np.all(w >= -1e-8)  # Numerical tolerance


def test_models_factory():
    """Factory returns correct instances."""
    assert isinstance(get_model("svi"), SVI)
    assert isinstance(get_model("SSVI"), SSVI)
    assert isinstance(get_model("EsSvI"), ESSVI)

    with pytest.raises(KeyError, match="invalid"):
        get_model("invalid")


def test_parametrization_abstract():
    """ABC cannot be instantiated."""
    from src.pysvi.models import Parametrization
    with pytest.raises(TypeError):
        Parametrization()


def test_factory_with_arbitrage_condition():
    """Factory passes arbitrage_condition to model instances."""
    svi_quasi = get_model("svi")
    assert svi_quasi.arbitrage_condition == ArbitrageFreedom.QUASI

    svi_bf = get_model("svi", ArbitrageFreedom.NO_BUTTERFLY)
    assert ArbitrageFreedom.NO_BUTTERFLY in svi_bf.arbitrage_condition

    ssvi_both = get_model("ssvi", ArbitrageFreedom.NO_BUTTERFLY | ArbitrageFreedom.NO_CALENDAR)
    assert ArbitrageFreedom.NO_BUTTERFLY in ssvi_both.arbitrage_condition
    assert ArbitrageFreedom.NO_CALENDAR in ssvi_both.arbitrage_condition


def test_svi_no_butterfly_calibration(atm_slice):
    """SVI with NO_BUTTERFLY produces non-negative density g(k)."""
    from src.pysvi.calibration import prepare_slice, calibrate_slice
    model = SVI(arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY)
    params = calibrate_slice(atm_slice, model)
    assert params is not None

    k, w_target, F = prepare_slice(atm_slice)
    k_check = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 500)
    w, dw, d2w = _svi_derivatives(
        k_check, params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    )
    g = (1.0 - k_check * dw / (2.0 * w)) ** 2 - (dw**2) / 4.0 * (1.0 / w + 0.25) + d2w / 2.0
    assert np.all(g >= -1e-6), f"Butterfly violation: min g = {g.min():.6f}"


def test_ssvi_no_butterfly_calibration(atm_slice):
    """SSVI with NO_BUTTERFLY produces non-negative density."""
    from src.pysvi.calibration import calibrate_slice
    model = SSVI(arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY)
    theta = float(np.nanmin(atm_slice["iv"] ** 2 * atm_slice["maturity"]))
    params = calibrate_slice(atm_slice, model, theta=theta)
    assert params is not None
    assert params["eta"] > 0


def test_svi_no_calendar_calibration(atm_slice):
    """SVI with NO_CALENDAR respects prior slice total variance."""
    from src.pysvi.calibration import prepare_slice, calibrate_slice
    k, w_target, F = prepare_slice(atm_slice)
    k_grid = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 200)

    # First calibrate a "prior" slice (quasi)
    model_prior = SVI()
    params_prior = calibrate_slice(atm_slice, model_prior)
    assert params_prior is not None
    w_prev = svi_total_variance(
        k_grid, params_prior["a"], params_prior["b"],
        params_prior["rho"], params_prior["m"], params_prior["sigma"]
    )

    # Now calibrate with NO_CALENDAR, passing w_prev
    model_cal = SVI(arbitrage_condition=ArbitrageFreedom.NO_CALENDAR)
    params_cal = model_cal.calibrate(k, w_target, w_prev=w_prev)
    assert params_cal is not None

    w_new = svi_total_variance(
        k_grid, params_cal["a"], params_cal["b"],
        params_cal["rho"], params_cal["m"], params_cal["sigma"]
    )
    # Total variance should not decrease vs prior
    assert np.all(w_new >= w_prev - 1e-6), "Calendar arbitrage violation"


def test_butterfly_penalty_helper():
    """Verify _butterfly_penalty returns 0 for well-behaved params."""
    k = np.linspace(-1, 1, 200)
    w, dw, d2w = _svi_derivatives(k, 0.01, 0.1, -0.5, 0.0, 0.2)
    assert _butterfly_penalty(k, w, dw, d2w) == 0.0


def test_calendar_penalty_helper():
    """Verify _calendar_penalty detects crossing total variances."""
    k_grid = np.linspace(-1, 1, 100)
    w_early = np.ones_like(k_grid) * 0.04   # T1 total var
    w_later = np.ones_like(k_grid) * 0.05   # T2 > T1
    assert _calendar_penalty(k_grid, w_later, w_early) == 0.0

    # Swap: later slice has less total var → violation
    assert _calendar_penalty(k_grid, w_early, w_later) > 0.0
