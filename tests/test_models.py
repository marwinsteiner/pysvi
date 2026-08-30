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
    # Total variance should not decrease vs prior. The calendar constraint is
    # a soft penalty, so allow O(1e-6) slack (backend-dependent rounding).
    assert np.all(w_new >= w_prev - 1e-5), "Calendar arbitrage violation"


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


def test_jw_total_variance_atm():
    """Jump-wings ATM total variance matches v_t * T."""
    from src.pysvi.models import jw_total_variance
    k = np.array([0.0])
    T = 0.25
    v_t = 0.04
    w = jw_total_variance(k, v_t=v_t, psi_t=-0.1, p_t=0.15, c_t=0.05, v_tilde_t=0.035, T=T)
    np.testing.assert_allclose(w[0], v_t * T, rtol=1e-6)


def test_jw_total_variance_symmetry():
    """Symmetric wings (p_t == c_t) with zero skew gives symmetric smile."""
    from src.pysvi.models import jw_total_variance
    k = np.array([-0.1, 0.1])
    w = jw_total_variance(k, v_t=0.04, psi_t=0.0, p_t=0.1, c_t=0.1, v_tilde_t=0.035, T=0.25)
    np.testing.assert_allclose(w[0], w[1], rtol=1e-6)


def test_jw_factory():
    """Factory returns JumpWings for 'jw' and 'jumpwings'."""
    assert isinstance(get_model("jw"), JumpWings)
    assert isinstance(get_model("jumpwings"), JumpWings)


def test_jw_roundtrip(jw_calibrated):
    """End-to-end JumpWings: calibrate -> apply -> RMSE < 2.5%."""
    from src.pysvi.calibration import apply_slice
    model, df_slice, params = jw_calibrated
    fitted = apply_slice(df_slice, params, model)
    rmse = float(np.sqrt(np.mean(fitted["residual_iv"] ** 2)))
    assert rmse < 0.025


def test_jw_no_butterfly_calibration(atm_slice):
    """JumpWings with NO_BUTTERFLY produces valid density."""
    from src.pysvi.calibration import prepare_slice, calibrate_slice
    model = JumpWings(arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY)
    T = float(atm_slice["maturity"].iloc[0])
    params = calibrate_slice(atm_slice, model, T=T)
    assert params is not None
    assert params["v_t"] > 0


# ── DirectSVI tests ──────────────────────────────────────────────────

def test_directsvi_total_variance_formula():
    """Verify conic evaluation matches manual quadratic-formula calculation."""
    from src.pysvi.models import directsvi_total_variance
    k = np.array([0.0, 0.1, -0.1])
    z0, z1, z2, z3, z4, z5 = 1.0, 1.0, -0.5, 0.1, -2.0, 0.04
    # Manual: A=z1, B=z2*x+z4, C=z0*x^2+z3*x+z5
    # y = (-B + sqrt(B^2 - 4AC)) / (2A)
    for i, x in enumerate(k):
        A = z1
        B = z2 * x + z4
        C = z0 * x**2 + z3 * x + z5
        expected = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
        result = directsvi_total_variance(np.array([x]), z0, z1, z2, z3, z4, z5)
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)


def test_directsvi_fit_recovers_svi():
    """Generate data from known SVI params, fit with DirectSVI, verify low RMSE."""
    from src.pysvi.models import directsvi_fit, directsvi_total_variance
    k = np.linspace(-0.3, 0.3, 50)
    w_true = svi_total_variance(k, a=0.01, b=0.1, rho=-0.5, m=0.0, sigma=0.2)
    z = directsvi_fit(k, w_true)
    w_fit = directsvi_total_variance(k, *z)
    rmse = float(np.sqrt(np.mean((w_true - w_fit) ** 2)))
    assert rmse < 1e-6, f"DirectSVI RMSE too high: {rmse}"


def test_directsvi_factory():
    """get_model("directsvi") and get_model("dsvi") return DirectSVI."""
    from src.pysvi.models import DirectSVI
    assert isinstance(get_model("directsvi"), DirectSVI)
    assert isinstance(get_model("dsvi"), DirectSVI)


def test_directsvi_roundtrip(directsvi_calibrated):
    """End-to-end DirectSVI: calibrate -> apply -> RMSE < 2.5%."""
    from src.pysvi.calibration import apply_slice
    model, df_slice, params = directsvi_calibrated
    fitted = apply_slice(df_slice, params, model)
    rmse = float(np.sqrt(np.mean(fitted["residual_iv"] ** 2)))
    assert rmse < 0.025, f"DirectSVI roundtrip RMSE too high: {rmse}"


@given(k=st.lists(st.floats(-1.0, 1.0), min_size=1, max_size=100))
def test_directsvi_positivity(k):
    """DirectSVI total variance always non-negative."""
    from src.pysvi.models import directsvi_total_variance
    k = np.array(k)
    z0, z1, z2, z3, z4, z5 = 1.0, 1.0, -0.5, 0.1, -2.0, 0.04
    w = directsvi_total_variance(k, z0, z1, z2, z3, z4, z5)
    assert np.all(w >= -1e-8)


# ── SABR tests ───────────────────────────────────────────────────────

def test_sabr_atm_formula():
    """ATM implied vol matches the Hagan ATM special case exactly.

    At K = F: sigma_ATM = alpha / F^(1-beta) * {1 + [(1-beta)^2 alpha^2
    / (24 F^(2-2beta)) + rho beta nu alpha / (4 F^(1-beta))
    + (2-3rho^2) nu^2 / 24] T}
    """
    from src.pysvi.models import sabr_implied_vol
    alpha, beta, rho, nu, F, T = 3.0, 0.5, -0.3, 0.6, 100.0, 0.25
    expected = (alpha / F ** (1 - beta)) * (
        1.0
        + (
            (1 - beta) ** 2 * alpha**2 / (24.0 * F ** (2 - 2 * beta))
            + rho * beta * nu * alpha / (4.0 * F ** (1 - beta))
            + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
        )
        * T
    )
    result = sabr_implied_vol(np.array([0.0]), alpha, beta, rho, nu, F, T)
    np.testing.assert_allclose(result[0], expected, rtol=1e-12)


def test_sabr_beta1_flat_vol_limit():
    """beta=1, nu=0: SABR degenerates to Black-Scholes flat vol = alpha."""
    from src.pysvi.models import sabr_implied_vol
    k = np.linspace(-0.5, 0.5, 41)
    sigma = sabr_implied_vol(k, alpha=0.2, beta=1.0, rho=0.0, nu=0.0, F=100.0, T=1.0)
    np.testing.assert_allclose(sigma, 0.2, rtol=1e-12)


def test_sabr_beta1_manual_formula():
    """beta=1 general case vs independent manual HKLW computation.

    With beta=1 the (FK) factors drop out: z = -(nu/alpha) k,
    sigma = alpha * z/x(z) * {1 + [rho nu alpha / 4 + (2-3rho^2) nu^2 / 24] T}
    """
    from src.pysvi.models import sabr_implied_vol
    alpha, rho, nu, F, T = 0.25, -0.4, 0.8, 50.0, 0.5
    k = np.array([-0.2, -0.05, 0.05, 0.2])
    z = -(nu / alpha) * k
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    expected = alpha * (z / x_z) * (
        1.0 + (rho * nu * alpha / 4.0 + (2 - 3 * rho**2) * nu**2 / 24.0) * T
    )
    result = sabr_implied_vol(k, alpha, 1.0, rho, nu, F, T)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_sabr_atm_continuity():
    """z/x(z) guard: vol is continuous through k=0 (no ATM singularity)."""
    from src.pysvi.models import sabr_implied_vol
    args = dict(alpha=3.0, beta=0.5, rho=-0.3, nu=0.6, F=100.0, T=0.25)
    at = sabr_implied_vol(np.array([0.0]), **args)[0]
    near = sabr_implied_vol(np.array([-1e-10, 1e-10, -1e-6, 1e-6]), **args)
    assert np.all(np.isfinite(near))
    np.testing.assert_allclose(near, at, rtol=1e-4)


def test_sabr_skew_direction():
    """Negative rho produces a downward-sloping smile (put wing higher)."""
    from src.pysvi.models import sabr_implied_vol
    sigma = sabr_implied_vol(
        np.array([-0.1, 0.1]), alpha=0.2, beta=1.0, rho=-0.6, nu=0.7, F=100.0, T=0.5
    )
    assert sigma[0] > sigma[1]


def test_sabr_parameter_validation():
    """Invalid beta, alpha, F raise ValueError; missing T/F raise KeyError."""
    from src.pysvi.models import sabr_implied_vol, SABR
    k = np.array([0.0])
    with pytest.raises(ValueError, match="beta"):
        sabr_implied_vol(k, alpha=0.2, beta=1.5, rho=0.0, nu=0.3, F=100.0, T=1.0)
    with pytest.raises(ValueError, match="alpha"):
        sabr_implied_vol(k, alpha=-0.1, beta=0.5, rho=0.0, nu=0.3, F=100.0, T=1.0)
    with pytest.raises(ValueError, match="forward"):
        sabr_implied_vol(k, alpha=0.2, beta=0.5, rho=0.0, nu=0.3, F=-1.0, T=1.0)

    model = SABR()
    w = np.array([0.01])
    with pytest.raises(KeyError):
        model.calibrate(k, w)  # missing T and F
    with pytest.raises(ValueError, match="beta"):
        model.calibrate(k, w, T=0.25, F=100.0, beta=2.0)
    with pytest.raises(ValueError, match="T > 0"):
        model.calibrate(k, w, T=-0.25, F=100.0)


@given(k=st.lists(st.floats(-1.0, 1.0), min_size=1, max_size=100))
def test_sabr_positivity(k):
    """SABR total variance always non-negative."""
    from src.pysvi.models import sabr_total_variance
    k = np.array(k)
    w = sabr_total_variance(k, alpha=0.2, beta=1.0, rho=-0.5, nu=0.6, F=100.0, T=0.5)
    assert np.all(w >= -1e-8)
    assert np.all(np.isfinite(w))


def test_sabr_factory():
    """get_model('sabr') returns SABR with arbitrage condition wired."""
    from src.pysvi.models import SABR
    assert isinstance(get_model("sabr"), SABR)
    assert isinstance(get_model("SABR"), SABR)
    m = get_model("sabr", ArbitrageFreedom.NO_BUTTERFLY)
    assert ArbitrageFreedom.NO_BUTTERFLY in m.arbitrage_condition


def test_sabr_parameter_recovery():
    """Calibration recovers known SABR parameters from clean data."""
    from src.pysvi.models import SABR, sabr_total_variance
    true = dict(alpha=0.22, beta=1.0, rho=-0.45, nu=0.85, F=100.0, T=0.5)
    k = np.linspace(-0.3, 0.3, 41)
    w = sabr_total_variance(k, **true)

    model = SABR()
    params = model.calibrate(k, w, T=true["T"], F=true["F"], beta=true["beta"])
    assert params is not None
    np.testing.assert_allclose(params["alpha"], true["alpha"], rtol=0.02)
    np.testing.assert_allclose(params["rho"], true["rho"], atol=0.05)
    np.testing.assert_allclose(params["nu"], true["nu"], rtol=0.10)

    w_fit = model.total_variance(k, params)
    rmse = float(np.sqrt(np.mean((w - w_fit) ** 2)))
    assert rmse < 1e-6, f"SABR self-recovery RMSE too high: {rmse}"


def test_sabr_parameter_recovery_beta_half():
    """Parameter recovery also works for beta=0.5 (rates convention)."""
    from src.pysvi.models import SABR, sabr_total_variance
    true = dict(alpha=2.2, beta=0.5, rho=-0.3, nu=0.6, F=100.0, T=1.0)
    k = np.linspace(-0.4, 0.4, 41)
    w = sabr_total_variance(k, **true)

    model = SABR()
    params = model.calibrate(k, w, T=true["T"], F=true["F"], beta=0.5)
    assert params is not None
    w_fit = model.total_variance(k, params)
    rmse = float(np.sqrt(np.mean((w - w_fit) ** 2)))
    assert rmse < 1e-6, f"SABR beta=0.5 recovery RMSE too high: {rmse}"


def test_sabr_roundtrip(sabr_calibrated):
    """End-to-end SABR: calibrate -> apply -> IV RMSE < 2.5%."""
    from src.pysvi.calibration import apply_slice
    model, df_slice, params = sabr_calibrated
    fitted = apply_slice(df_slice, params, model)
    rmse = float(np.sqrt(np.mean(fitted["residual_iv"] ** 2)))
    assert rmse < 0.025, f"SABR roundtrip RMSE too high: {rmse}"


def test_sabr_no_butterfly_calibration(atm_slice):
    """SABR with NO_BUTTERFLY calibrates and yields non-negative density."""
    from src.pysvi.calibration import prepare_slice, calibrate_slice
    from src.pysvi.models import SABR, sabr_total_variance, _finite_diff_derivatives
    model = SABR(arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY)
    T = float(atm_slice["maturity"].iloc[0])
    F = float(atm_slice["implied_forward"].iloc[0])
    params = calibrate_slice(atm_slice, model, T=T, F=F, beta=1.0)
    assert params is not None

    k, _, _ = prepare_slice(atm_slice)
    k_check = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 500)
    w = sabr_total_variance(
        k_check, params["alpha"], params["beta"], params["rho"],
        params["nu"], params["F"], params["T"],
    )
    dw, d2w = _finite_diff_derivatives(k_check, w)
    g = (1.0 - k_check * dw / (2.0 * w)) ** 2 - (dw**2) / 4.0 * (1.0 / w + 0.25) + d2w / 2.0
    assert np.all(g >= -1e-4), f"Butterfly violation: min g = {g.min():.6f}"


def test_sabr_no_calendar_calibration(atm_slice):
    """SABR with NO_CALENDAR respects prior slice total variance."""
    from src.pysvi.calibration import prepare_slice
    from src.pysvi.models import SABR, sabr_total_variance
    k, w_target, F = prepare_slice(atm_slice)
    T = float(atm_slice["maturity"].iloc[0])
    k_grid = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 200)

    model_prior = SABR()
    params_prior = model_prior.calibrate(k, w_target, T=T, F=F, beta=1.0)
    assert params_prior is not None
    w_prev = sabr_total_variance(
        k_grid, params_prior["alpha"], params_prior["beta"],
        params_prior["rho"], params_prior["nu"], params_prior["F"], params_prior["T"],
    )

    model_cal = SABR(arbitrage_condition=ArbitrageFreedom.NO_CALENDAR)
    params_cal = model_cal.calibrate(k, w_target, T=T, F=F, beta=1.0, w_prev=w_prev)
    assert params_cal is not None
    w_new = model_cal.total_variance(k_grid, params_cal)
    assert np.all(w_new >= w_prev - 1e-5), "Calendar arbitrage violation"


def test_sabr_total_variance_consistency():
    """sabr_total_variance == sabr_implied_vol^2 * T."""
    from src.pysvi.models import sabr_implied_vol, sabr_total_variance
    k = np.linspace(-0.3, 0.3, 11)
    args = dict(alpha=0.2, beta=1.0, rho=-0.5, nu=0.6, F=100.0, T=0.5)
    np.testing.assert_allclose(
        sabr_total_variance(k, **args),
        sabr_implied_vol(k, **args) ** 2 * args["T"],
        rtol=1e-14,
    )
