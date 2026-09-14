"""Parameter identifiability and uncertainty (issue #25)."""

import numpy as np
import pytest

from src.pysvi.identifiability import (
    condition_number, identifiability_report, parameter_uncertainty,
)
from src.pysvi.models import (
    ESSVI, JumpWings, NaturalSVI, Parametrization, SABR, SSVI, SVI,
    svi_total_variance,
)

TRUE = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}


def _noisy_smile(lo, hi, n, seed=3, noise=0.005):
    rng = np.random.default_rng(seed)
    k = np.linspace(lo, hi, n)
    w = svi_total_variance(k, **TRUE) * (1.0 + noise * rng.standard_normal(n))
    return k, w


# ── Jacobians ────────────────────────────────────────────────────────

@pytest.mark.parametrize("model,params", [
    (SVI(), TRUE),
    (NaturalSVI(), {"delta": 0.005, "mu": 0.02, "rho": -0.5,
                    "omega": 0.04, "zeta": 1.8}),
    (SSVI(), {"theta": 0.02, "rho": -0.5, "eta": 1.3}),
], ids=["svi", "natural", "ssvi"])
def test_analytic_jacobian_matches_fd(model, params):
    """The analytic overrides agree with the finite-difference base."""
    k = np.linspace(-0.3, 0.3, 11)
    J = model.param_jacobian(k, params)
    J_fd = Parametrization.param_jacobian(model, k, params)
    np.testing.assert_allclose(J, J_fd, atol=1e-6)


@pytest.mark.parametrize("model,params", [
    (ESSVI(), {"theta": 0.02, "theta_ref": 0.02, "rho0": -0.3,
               "rho1": -0.2, "alpha": 0.5, "eta": 1.3, "rho_theta": -0.5}),
    (JumpWings(), {"v_t": 0.04, "psi_t": -0.1, "p_t": 0.15, "c_t": 0.05,
                   "v_tilde_t": 0.035, "T": 0.5}),
    (SABR(), {"alpha": 0.2, "beta": 0.5, "rho": -0.4, "nu": 0.6,
              "F": 100.0, "T": 0.5}),
], ids=["essvi", "jw", "sabr"])
def test_fd_jacobian_shape_and_finiteness(model, params):
    """FD fallback produces a finite n x p Jacobian for every model."""
    k = np.linspace(-0.3, 0.3, 11)
    J = model.param_jacobian(k, params)
    assert J.shape == (11, len(model.free_params))
    assert np.all(np.isfinite(J))


def test_condition_number_rank_deficient_is_huge():
    J = np.column_stack([np.ones(5), np.ones(5)])  # collinear columns
    assert condition_number(J) > 1e12  # inf up to SVD rounding
    assert condition_number(np.column_stack([np.ones(5), np.zeros(5)])) == float("inf")


# ── Acceptance: narrow vs wide strike ranges ─────────────────────────

def test_narrow_strikes_flag_poor_identification():
    """Acceptance (issue #25): tiny IV RMSE, yet the wing parameters are
    flagged as poorly identified on a narrow strike range."""
    k, w = _noisy_smile(-0.05, 0.05, 15)
    model = SVI()
    params = model.calibrate(k, w, initialization="multi_start")
    report = identifiability_report(model, params, k, w)
    assert report.rmse_w < 1e-3          # the smile fits fine
    assert not report.ok                 # the parameters do not
    assert len(report.poorly_identified) >= 2
    assert report.degenerate_pairs       # near-degenerate directions
    assert report.condition_number > 1e3
    text = str(report)
    assert "poorly identified" in text and "ATTENTION" in text


def test_uncertainties_shrink_with_strike_range():
    """Acceptance (issue #25): widening the quoted range shrinks the
    parameter standard errors."""
    model = SVI()
    k1, w1 = _noisy_smile(-0.05, 0.05, 31)
    k2, w2 = _noisy_smile(-0.5, 0.5, 31)
    p1 = model.calibrate(k1, w1, initialization="multi_start")
    p2 = model.calibrate(k2, w2, initialization="multi_start")
    u1 = parameter_uncertainty(model, p1, k1, w1)
    u2 = parameter_uncertainty(model, p2, k2, w2)
    # every parameter is at least 10x better determined on the wide range
    for name, se1, se2 in zip(u1.names, u1.std_errors, u2.std_errors):
        assert se2 < se1 / 10.0, name


def test_wide_range_well_identified_values():
    """On a wide range the report's flags clear and errors are small
    relative to the parameter magnitudes."""
    k, w = _noisy_smile(-0.5, 0.5, 41, noise=0.002)
    model = SVI()
    params = model.calibrate(k, w, initialization="multi_start")
    report = identifiability_report(model, params, k, w)
    assert not report.poorly_identified
    u = report.uncertainty
    assert u.dof == 41 - 5
    for name, v, se in zip(u.names, u.values, u.std_errors):
        if name != "m":  # m ~ 0.01: tiny magnitude, rel error unstable
            assert se < 0.5 * abs(v), name


def test_under_determined_fit_is_inf():
    """n <= p leaves no residual degrees of freedom: every std error inf."""
    k = np.linspace(-0.1, 0.1, 5)
    w = svi_total_variance(k, **TRUE)
    u = parameter_uncertainty(SVI(), TRUE, k, w)
    assert u.dof == 0
    assert all(se == float("inf") for se in u.std_errors)


def test_report_correlation_is_symmetric_unit_diagonal():
    k, w = _noisy_smile(-0.4, 0.4, 41)
    model = SVI()
    params = model.calibrate(k, w, initialization="multi_start")
    u = parameter_uncertainty(model, params, k, w)
    np.testing.assert_allclose(u.correlation, u.correlation.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(u.correlation), 1.0, atol=1e-9)
