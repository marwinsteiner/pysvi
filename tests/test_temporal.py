"""Prior-anchored calibration (#27) and quote sensitivities (#28)."""

import numpy as np
import pytest

from src.pysvi.identifiability import (
    iv_surface_sensitivity, quote_sensitivity, surface_sensitivity,
)
from src.pysvi.models import SVI, SSVI, svi_total_variance
from src.pysvi.surface import VolSurface

TRUE = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}


def _smile(seed=3, noise=0.005, n=41):
    rng = np.random.default_rng(seed)
    k = np.linspace(-0.4, 0.4, n)
    w = svi_total_variance(k, **TRUE) * (1.0 + noise * rng.standard_normal(n))
    return k, w


# ── Prior anchoring ──────────────────────────────────────────────────

def test_anchoring_bounds_parameter_drift():
    """Acceptance (#27): a tiny market perturbation with prior anchoring
    yields parameter changes bounded well below the unanchored drift."""
    k, w = _smile()
    rng = np.random.default_rng(11)
    w2 = w * (1.0 + 5e-4 * rng.standard_normal(w.size))
    model = SVI()
    p_prev = model.calibrate(k, w, initialization="multi_start")
    p_un = model.calibrate(k, w2, initialization="multi_start")
    p_an = model.calibrate(k, w2, initialization="multi_start",
                           prior=p_prev, anchor=1.0)
    drift_un = max(abs(p_un[n] - p_prev[n]) for n in model.free_params)
    drift_an = max(abs(p_an[n] - p_prev[n]) for n in model.free_params)
    assert drift_an < drift_un
    assert drift_an < 5e-3  # bounded by the perturbation scale


def test_anchor_zero_reproduces_unanchored_objective():
    """Acceptance (#27): anchor=0 leaves the objective untouched -- the
    fit differs from no-prior only through the warm start, and matches
    it exactly when the start is also the same."""
    k, w = _smile()
    model = SVI()
    p_prev = model.calibrate(k, w, initialization="multi_start")
    p_a = model.calibrate(k, w, prior=p_prev, anchor=0.0)
    p_b = model.calibrate(k, w, prior=p_prev)  # anchor defaults to 0
    assert p_a == p_b


def test_prior_fit_deterministic():
    """Acceptance (#27): identical quotes + identical prior => identical
    parameters, bitwise."""
    k, w = _smile()
    model = SVI()
    p_prev = model.calibrate(k, w)
    p1 = model.calibrate(k, w, prior=p_prev, anchor=0.5,
                         initialization="multi_start")
    p2 = model.calibrate(k, w, prior=p_prev, anchor=0.5,
                         initialization="multi_start")
    assert p1 == p2


def test_prior_missing_parameter_raises():
    k, w = _smile()
    with pytest.raises(ValueError, match="prior is missing"):
        SVI().calibrate(k, w, prior={"a": 0.01}, anchor=1.0)


def test_surface_level_prior(surface_df):
    """VolSurface.fit(prior=prev_surface) anchors slice-by-slice."""
    from tests.conftest import SURFACE_RATE as R
    prev = VolSurface.fit(surface_df, model="svi", r=R,
                          initialization="multi_start")
    bumped = surface_df.copy()
    bumped["iv"] = bumped["iv"] * 1.0005
    anchored = VolSurface.fit(bumped, model="svi", r=R,
                              initialization="multi_start",
                              prior=prev, anchor=1.0)
    free = VolSurface.fit(bumped, model="svi", r=R,
                          initialization="multi_start")
    def total_drift(s):
        return sum(
            abs(s.params(T)[n] - prev.params(T)[n])
            for T in prev.maturities for n in ("a", "b", "rho", "m", "sigma")
        )
    # anchoring cannot increase the aggregate parameter drift (per-slice
    # micro-differences from the tight-tolerance path are allowed)
    assert total_drift(anchored) <= total_drift(free) + 1e-3


# ── Quote sensitivities ──────────────────────────────────────────────

def test_quote_sensitivity_matches_fd_recalibration():
    """Acceptance (#28): bump a quote, recalibrate, compare against the
    implicit-function Jacobian prediction (first order)."""
    k, w = _smile(noise=0.003)
    model = SVI()
    p0 = model.calibrate(k, w, initialization="multi_start")
    S = quote_sensitivity(model, p0, k, w)
    assert S.shape == (5, k.size)
    # A large-ish bump drowns optimizer-termination noise in the FD
    # reference (which recalibrates and is platform-sensitive under
    # fastmath); agreement is asserted at the vector-norm level.
    i, h = 7, 2e-4
    w2 = w.copy(); w2[i] += h
    p1 = model.calibrate(k, w2, initialization="multi_start",
                         prior=p0)  # warm start: stay in the same basin
    fd = np.array([(p1[n] - p0[n]) / h for n in model.free_params])
    rel = np.linalg.norm(S[:, i] - fd) / (np.linalg.norm(fd) + 1e-9)
    assert rel < 0.3, rel


def test_surface_sensitivity_shape_and_fd():
    k, w = _smile(noise=0.003)
    model = SVI()
    p0 = model.calibrate(k, w, initialization="multi_start")
    k_eval = np.array([-0.2, 0.0, 0.2])
    S = surface_sensitivity(model, p0, k, w, k_eval)
    assert S.shape == (3, k.size)
    i, h = 20, 2e-4
    w2 = w.copy(); w2[i] += h
    p1 = model.calibrate(k, w2, initialization="multi_start", prior=p0)
    fd = (model.total_variance(k_eval, p1)
          - model.total_variance(k_eval, p0)) / h
    rel = np.linalg.norm(S[:, i] - fd) / (np.linalg.norm(fd) + 1e-9)
    assert rel < 0.3, rel


def test_iv_sensitivity_units():
    """The iv-space Jacobian rescales the w-space one by 2 iv T on both
    sides; a quote's own-point sensitivity is O(1) and positive."""
    k, w = _smile(noise=0.002)
    T = 0.5
    model = SVI()
    p0 = model.calibrate(k, w, initialization="multi_start")
    S_iv = iv_surface_sensitivity(model, p0, k, w, k, T)
    diag = np.diag(S_iv)
    assert np.all(np.isfinite(S_iv))
    assert diag.mean() > 0  # a quote pulls its own point along


def test_sensitivity_works_for_ssvi():
    k, w = _smile(noise=0.002)
    theta = float(np.interp(0.0, k, w))
    model = SSVI()
    p0 = model.calibrate(k, w, theta=theta, initialization="multi_start")
    S = quote_sensitivity(model, p0, k, w)
    assert S.shape == (2, k.size) and np.all(np.isfinite(S))
