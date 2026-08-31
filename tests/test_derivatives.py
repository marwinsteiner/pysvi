"""Public derivative and density API: dw_dk, d2w_dk2, density."""

import numpy as np
import pytest

from src.pysvi.models import (
    SVI, NaturalSVI, SSVI, ESSVI, JumpWings, SABR, DirectSVI, Parametrization,
    ArbitrageFreedom,
)
from src.pysvi.calibration import calibrate_slice

_K = np.linspace(-0.5, 0.5, 41)

# (model instance, params) for every parametrization
_ANALYTIC_CASES = [
    (SVI(), {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}),
    (NaturalSVI(), {"delta": 0.005, "mu": 0.02, "rho": -0.5, "omega": 0.04, "zeta": 1.8}),
    (SSVI(), {"rho": -0.5, "eta": 1.2, "theta": 0.02}),
    (ESSVI(), {"rho0": 0.0, "rho1": -0.5, "alpha": 0.5, "eta": 1.0,
               "theta": 0.02, "theta_ref": 0.02}),
    (JumpWings(), {"v_t": 0.04, "psi_t": -0.1, "p_t": 0.15, "c_t": 0.05,
                   "v_tilde_t": 0.035, "T": 0.25}),
]
_FD_CASES = [
    (SABR(), {"alpha": 0.2, "beta": 1.0, "rho": -0.4, "nu": 0.6,
              "F": 100.0, "T": 0.5}),
    (DirectSVI(), {"z0": 1.0, "z1": 1.0, "z2": -0.5, "z3": 0.1,
                   "z4": -2.0, "z5": 0.04}),
]


@pytest.mark.parametrize(
    "model,params", _ANALYTIC_CASES,
    ids=[type(m).__name__ for m, _ in _ANALYTIC_CASES],
)
def test_analytic_dw_dk_matches_numerical(model, params):
    """Analytic w'(k) matches central finite differences to 1e-8."""
    dw_analytic = model.dw_dk(_K, params)
    # Unbound base-class call forces the finite-difference implementation
    _, dw_fd, _ = Parametrization.derivatives(model, _K, params)
    np.testing.assert_allclose(dw_analytic, dw_fd, atol=1e-8)


@pytest.mark.parametrize(
    "model,params", _ANALYTIC_CASES,
    ids=[type(m).__name__ for m, _ in _ANALYTIC_CASES],
)
def test_analytic_d2w_dk2_matches_numerical(model, params):
    """Analytic w''(k) matches finite differences within FD roundoff (~1e-6)."""
    d2w_analytic = model.d2w_dk2(_K, params)
    _, _, d2w_fd = Parametrization.derivatives(model, _K, params)
    np.testing.assert_allclose(d2w_analytic, d2w_fd, atol=1e-6)


@pytest.mark.parametrize(
    "model,params", _ANALYTIC_CASES + _FD_CASES,
    ids=[type(m).__name__ for m, _ in _ANALYTIC_CASES + _FD_CASES],
)
def test_derivatives_consistent_with_wrappers(model, params):
    """derivatives() agrees with total_variance/dw_dk/d2w_dk2 wrappers."""
    w, dw, d2w = model.derivatives(_K, params)
    np.testing.assert_allclose(w, model.total_variance(_K, params), rtol=1e-12)
    np.testing.assert_allclose(dw, model.dw_dk(_K, params), rtol=1e-12)
    np.testing.assert_allclose(d2w, model.d2w_dk2(_K, params), rtol=1e-12)


def test_wing_slopes_analytic_models():
    """Closed-form wing slopes match large-k measured slopes."""
    far = np.array([-60.0, 60.0])
    for model, params in _ANALYTIC_CASES:
        slopes = model.wing_slopes(params)
        assert slopes is not None
        dw = model.dw_dk(far, params)
        np.testing.assert_allclose(slopes[0], -dw[0], rtol=1e-3)
        np.testing.assert_allclose(slopes[1], dw[1], rtol=1e-3)
    for model, params in _FD_CASES:
        assert model.wing_slopes(params) is None


def test_essvi_rho_theta_only_params():
    """A params dict carrying rho_theta directly needs no rho0/rho1/alpha."""
    model = ESSVI()
    params = {"theta": 0.02, "eta": 1.0, "rho_theta": -0.5}
    w = model.total_variance(_K, params)
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(model.density(_K, params)))
    assert model.wing_slopes(params) is not None


@pytest.mark.parametrize(
    "model,params", _ANALYTIC_CASES + _FD_CASES,
    ids=[type(m).__name__ for m, _ in _ANALYTIC_CASES + _FD_CASES],
)
def test_density_finite(model, params):
    """g(k) evaluates finite for every parametrization."""
    g = model.density(_K, params)
    assert g.shape == _K.shape
    assert np.all(np.isfinite(g))


def test_svi_dw_dk_closed_form():
    """SVI w'(k) matches the closed form b(rho + z/sqrt(z^2 + sigma^2))."""
    model = SVI()
    a, b, rho, m, sigma = 0.01, 0.12, -0.6, 0.01, 0.25
    params = {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}
    z = _K - m
    expected = b * (rho + z / np.sqrt(z * z + sigma * sigma))
    np.testing.assert_allclose(model.dw_dk(_K, params), expected, rtol=1e-12)


def test_sabr_flat_vol_derivatives_vanish():
    """beta=1, nu=0 SABR is flat: w'(k) = w''(k) = 0 up to FD noise."""
    model = SABR()
    params = {"alpha": 0.2, "beta": 1.0, "rho": 0.0, "nu": 0.0,
              "F": 100.0, "T": 1.0}
    np.testing.assert_allclose(model.dw_dk(_K, params), 0.0, atol=1e-10)
    np.testing.assert_allclose(model.d2w_dk2(_K, params), 0.0, atol=1e-5)


def test_density_nonnegative_on_no_butterfly_fit(atm_slice):
    """A NO_BUTTERFLY-calibrated SVI slice has g(k) >= 0 on the data range."""
    model = SVI(arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY)
    params = calibrate_slice(atm_slice, model)
    assert params is not None
    k = np.linspace(-0.7, 0.7, 500)
    g = model.density(k, params)
    assert np.all(g >= -1e-6), f"negative density: min g = {g.min():.6f}"


def test_derivatives_backend_parity(backend_mode):
    """Public derivatives agree across numba/NumPy backends."""
    model = SVI()
    params = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}
    dw = model.dw_dk(_K, params)
    d2w = model.d2w_dk2(_K, params)
    z = _K - params["m"]
    r = np.sqrt(z * z + params["sigma"] ** 2)
    np.testing.assert_allclose(dw, params["b"] * (params["rho"] + z / r), rtol=1e-10)
    np.testing.assert_allclose(
        d2w, params["b"] * params["sigma"] ** 2 / r**3, rtol=1e-10
    )


def test_jw_flat_slice_derivatives():
    """Degenerate jump-wings slice (b ~ 0) has zero derivatives."""
    model = JumpWings()
    params = {"v_t": 0.04, "psi_t": 0.0, "p_t": 0.0, "c_t": 0.0,
              "v_tilde_t": 0.04, "T": 0.25}
    np.testing.assert_array_equal(model.dw_dk(_K, params), 0.0)
    np.testing.assert_array_equal(model.d2w_dk2(_K, params), 0.0)
