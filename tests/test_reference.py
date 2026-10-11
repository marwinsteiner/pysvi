"""Reference-value anchors and edge-of-domain battery (issue #14).

Two layers of protection:

* Hand-derived closed-form anchors: exact identities of each
  parametrization at special points, derived independently of the
  implementation (provenance in each test docstring).
* An edge-of-domain battery near parameter boundaries, run under both
  backends, including numba/NumPy parity at the numerical decision
  boundaries the arbitrage diagnostics depend on.
"""

import numpy as np
import pytest

from src.pysvi import _kernels as K
from src.pysvi.models import (
    SVI, NaturalSVI, SSVI, ESSVI, JumpWings, SABR,
    essvi_total_variance, jw_total_variance, natural_total_variance,
    sabr_implied_vol, ssvi_total_variance, svi_total_variance,
    numba_available, use_numba,
)
from src.pysvi.diagnostics import check_slice_arbitrage

needs_numba = pytest.mark.skipif(not numba_available(), reason="numba not installed")


# ── Hand-derived closed-form anchors ─────────────────────────────────

def test_svi_anchor_at_vertex_shift():
    """At k = m: w = a + b*sigma (z = 0 kills the rho term; sqrt(sigma^2) = sigma)."""
    a, b, rho, m, sigma = 0.013, 0.21, -0.7, 0.035, 0.4
    w = svi_total_variance(np.array([m]), a, b, rho, m, sigma)
    np.testing.assert_allclose(w[0], a + b * sigma, rtol=1e-15)


def test_svi_anchor_wing_slopes():
    """Far-wing slope of w tends to b(1 +/- rho) (Gatheral 2004 asymptotics)."""
    a, b, rho, m, sigma = 0.01, 0.3, -0.4, 0.0, 0.2
    k = np.array([-80.0, 80.0])
    w = svi_total_variance(k + 1.0, a, b, rho, m, sigma) - svi_total_variance(k, a, b, rho, m, sigma)
    np.testing.assert_allclose(-w[0], b * (1 - rho), rtol=1e-3)
    np.testing.assert_allclose(w[1], b * (1 + rho), rtol=1e-3)


def test_ssvi_anchor_atm():
    """At k = 0: w = theta exactly (term1 = 1, sqrt(rho^2 + 1 - rho^2) = 1)."""
    theta, rho, phi = 0.04, -0.65, 1.7
    w = ssvi_total_variance(np.array([0.0]), theta, rho, phi)
    np.testing.assert_allclose(w[0], theta, rtol=1e-15)
    w = essvi_total_variance(np.array([0.0]), theta, rho, phi)
    np.testing.assert_allclose(w[0], theta, rtol=1e-15)


def test_natural_anchor_at_mu():
    """At k = mu: w = Delta + omega (bracket = 1 + 0 + sqrt(rho^2 + 1 - rho^2) = 2)."""
    delta, mu, rho, omega, zeta = 0.007, 0.03, -0.55, 0.05, 1.4
    w = natural_total_variance(np.array([mu]), delta, mu, rho, omega, zeta)
    np.testing.assert_allclose(w[0], delta + omega, rtol=1e-15)


def test_jw_anchor_atm():
    """At k = 0: w = v_t * T by construction of the jump-wings map."""
    T = 0.7
    w = jw_total_variance(np.array([0.0]), 0.05, -0.2, 0.3, 0.1, 0.04, T)
    np.testing.assert_allclose(w[0], 0.05 * T, rtol=1e-9)


def test_sabr_anchor_beta_zero_atm():
    """Hagan ATM with beta = 0: sigma = alpha/F * (1 + [alpha^2/(24 F^2) + (2-3rho^2)nu^2/24] T).

    Hand-substitution of beta = 0 into the HKLW ATM formula (the rho
    term carries a factor beta and vanishes).
    """
    alpha, rho, nu, F, T = 5.0, -0.3, 0.4, 100.0, 1.5
    expected = alpha / F * (
        1.0 + (alpha**2 / (24.0 * F**2) + (2.0 - 3.0 * rho**2) * nu**2 / 24.0) * T
    )
    got = sabr_implied_vol(np.array([0.0]), alpha, 0.0, rho, nu, F, T)
    np.testing.assert_allclose(got[0], expected, rtol=1e-12)


# ── Pinned regression tables ─────────────────────────────────────────
# Values pinned at v1.0.0 from the NumPy reference implementation and
# cross-checked against the closed-form anchors above; they guard
# against silent numerical drift in either backend.

_K5 = np.array([-0.4, -0.1, 0.0, 0.1, 0.4])

_PINNED = [
    (svi_total_variance, (0.01, 0.12, -0.6, 0.01, 0.25),
     [0.097144994577006, 0.050695600680994, 0.040743990407672,
      0.035404792613407, 0.037509927145122]),
    (ssvi_total_variance, (0.02, -0.5, 1.3),
     [0.025980582946942, 0.021359341716464, 0.020000000000000,
      0.018767536832952, 0.016062563131083]),
    (natural_total_variance, (0.005, 0.02, -0.5, 0.04, 1.8),
     [0.063072528574341, 0.049633593393136, 0.045729545895773,
      0.042286836358552, 0.035867128508033]),
]


@pytest.mark.parametrize("fn,params,expected", _PINNED, ids=["svi", "ssvi", "natural"])
def test_pinned_regression_tables(fn, params, expected):
    np.testing.assert_allclose(fn(_K5, *params), np.array(expected), rtol=1e-10)


# ── Edge-of-domain battery ───────────────────────────────────────────

_EDGE_CASES = [
    ("svi rho->+0.999", lambda k: svi_total_variance(k, 0.01, 0.3, 0.999, 0.0, 0.2)),
    ("svi rho->-0.999", lambda k: svi_total_variance(k, 0.01, 0.3, -0.999, 0.0, 0.2)),
    ("svi sigma->1e-8", lambda k: svi_total_variance(k, 0.01, 0.1, -0.5, 0.0, 1e-8)),
    ("ssvi eta tiny", lambda k: ssvi_total_variance(k, 0.02, -0.9, 1e-8)),
    ("natural zeta tiny", lambda k: natural_total_variance(k, 0.01, 0.0, -0.5, 0.04, 1e-6)),
    ("jw T tiny", lambda k: jw_total_variance(k, 0.04, -0.1, 0.15, 0.05, 0.035, 1e-4)),
    ("jw T huge", lambda k: jw_total_variance(k, 0.04, -0.1, 0.15, 0.05, 0.035, 30.0)),
    ("sabr T tiny", lambda k: sabr_implied_vol(k, 0.2, 1.0, -0.4, 0.6, 100.0, 1e-4)),
    ("sabr T huge", lambda k: sabr_implied_vol(k, 0.2, 1.0, -0.4, 0.6, 100.0, 30.0)),
    ("sabr nu zero", lambda k: sabr_implied_vol(k, 0.2, 0.5, 0.0, 0.0, 100.0, 1.0)),
]


@pytest.mark.parametrize("label,fn", _EDGE_CASES, ids=[c[0] for c in _EDGE_CASES])
def test_edge_battery_finite(backend_mode, label, fn):
    """Boundary parameters produce finite, non-negative values on both
    backends, including at the |k| = 10 clipping bound."""
    k = np.array([-10.0, -1.0, -1e-10, 0.0, 1e-10, 1.0, 10.0])
    values = fn(k)
    assert np.all(np.isfinite(values)), label
    assert np.all(values >= -1e-10), label


def test_degenerate_calibration_returns_none_not_raise():
    """Garbage targets fail gracefully (None), never with an exception."""
    k = np.linspace(-0.2, 0.2, 11)
    w = np.full(11, 1e-15)
    for model, kwargs in [
        (SVI(), {}),
        (NaturalSVI(), {}),
        (SSVI(), {"theta": 1e-15}),
        (JumpWings(), {"T": 0.25}),
    ]:
        result = model.calibrate(k, w, **kwargs)
        assert result is None or isinstance(result, dict)


# ── fastmath decision-boundary parity ────────────────────────────────

_BOUNDARY_KERNELS = [
    ("svi_w near rho boundary", "svi_w", (0.005, 0.2, 0.9989, 0.0, 1e-6)),
    ("svi_derivs near sigma zero", "svi_derivs", (0.01, 0.1, -0.5, 0.0, 1e-7)),
    ("ssvi_w near rho boundary", "ssvi_w", (0.02, -0.9989, 2.5)),
    ("sabr_vol near z zero", "sabr_vol", (0.22, 0.5, -0.45, 1e-9, 100.0, 0.5)),
]


@needs_numba
@pytest.mark.parametrize("label,name,params", _BOUNDARY_KERNELS, ids=[b[0] for b in _BOUNDARY_KERNELS])
def test_fastmath_parity_at_boundaries(label, name, params):
    """Jitted (fastmath) and NumPy kernels agree at the numerical
    boundaries the diagnostics depend on (w->0, rho->+-1, sigma->0, z->0)."""
    k = np.array([-2.0, -1e-9, 0.0, 1e-9, 2.0])
    plain = K._PLAIN[name](k, *params)
    jitted = K._JITTED[name](k, *params)
    if isinstance(plain, tuple):
        for p, j in zip(plain, jitted):
            np.testing.assert_allclose(p, j, rtol=1e-8, atol=1e-14)
    else:
        np.testing.assert_allclose(plain, jitted, rtol=1e-8, atol=1e-14)


@needs_numba
def test_fastmath_cannot_flip_arbitrage_verdict():
    """A near-boundary slice gets the same arbitrage verdict on both backends."""
    model = SVI()
    params = {"a": 0.001, "b": 0.35, "rho": -0.995, "m": 0.0, "sigma": 1e-5}
    prev = K.numba_enabled()
    try:
        use_numba(False)
        rep_np = check_slice_arbitrage(model, params)
        use_numba(True)
        rep_nb = check_slice_arbitrage(model, params)
    finally:
        use_numba(prev)
    assert rep_np.butterfly_free == rep_nb.butterfly_free
    assert rep_np.lee_free == rep_nb.lee_free
    assert rep_np.ok == rep_nb.ok
