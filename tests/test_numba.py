"""Numba backend: parity with NumPy, toggle behaviour, env-var control."""

import os
import subprocess
import sys

import numpy as np
import pytest

from src.pysvi import _kernels as K
from src.pysvi.models import use_numba, numba_available, SVI
from src.pysvi.calibration import prepare_slice

needs_numba = pytest.mark.skipif(
    not numba_available(), reason="numba not installed"
)

_k = np.linspace(-0.4, 0.4, 41)
_kg = np.linspace(-0.9, 0.9, 200)
_w = K.svi_w(_k, 0.01, 0.12, -0.6, 0.01, 0.25) + 1e-4
_wg = K.svi_w(_kg, 0.01, 0.12, -0.6, 0.01, 0.25)
_empty = np.empty(0)

# name -> args exercising the kernel (grid-penalty branches enabled where
# applicable, calendar with a real w_prev)
_PARITY_CASES = {
    "svi_w": (_k, 0.01, 0.12, -0.6, 0.01, 0.25),
    "ssvi_w": (_k, 0.02, -0.5, 1.3),
    "essvi_w": (_k, 0.02, -0.4, 1.1),
    "jw_w": (_k, 0.04, -0.1, 0.15, 0.05, 0.035, 0.25),
    "sabr_vol": (_k, 0.22, 0.5, -0.45, 0.85, 100.0, 0.5),
    "directsvi_w": (_k, 1.0, 1.0, -0.5, 0.1, -2.0, 0.04),
    "svi_derivs": (_kg, 0.01, 0.12, -0.6, 0.01, 0.25),
    "ssvi_derivs": (_kg, 0.02, -0.5, 1.3),
    "butterfly": (_kg, _wg, np.gradient(_wg, _kg), np.gradient(np.gradient(_wg, _kg), _kg)),
    "density_g": (_kg, _wg, np.gradient(_wg, _kg), np.gradient(np.gradient(_wg, _kg), _kg)),
    "jw_convert": (0.04, -0.1, 0.15, 0.05, 0.035, 0.25),
    "natural_w": (_k, 0.005, 0.02, -0.5, 0.04, 1.8),
    "natural_convert": (0.005, 0.02, -0.5, 0.04, 1.8),
    "black_call": (0.05, 0.012),
    "loss_value": (_k, _w * 1.02, _w, 2, _empty, _empty, _empty, 2, 0.01),
    "residuals": (_k, _w * 1.02, _w, 3, np.ones_like(_w), _empty, _empty),
    "natural_obj": (np.array([0.005, 0.0, -0.4, 0.05, 1.5]), _k, _w, _kg, _wg * 0.9, True, True, True, 0, _empty, _empty, _empty, 0, 1.0),
    "calendar": (_wg, _wg * 1.01),
    "finite_diff": (_kg, _wg),
    "svi_obj": (np.array([0.011, 0.1, -0.5, 0.0, 0.2]), _k, _w, _kg, _wg * 0.9, True, True, True, 0, _empty, _empty, _empty, 0, 1.0),
    "ssvi_obj": (np.array([-0.5, 1.2]), _k, _w, 0.02, _kg, _wg * 0.9, True, True, True, 0, _empty, _empty, _empty, 0, 1.0),
    "essvi_obj": (np.array([0.0, -0.5, 0.5, 1.0]), _k, _w, 0.02, 0.02, _kg, _wg * 0.9, True, True, True, 0, _empty, _empty, _empty, 0, 1.0),
    "jw_obj": (np.array([0.04, -0.1, 0.1, 0.1, 0.035]), _k, _w, 0.25, _kg, _wg * 0.9, True, True, True, 0, _empty, _empty, _empty, 0, 1.0),
    "sabr_obj": (np.array([0.2, -0.4, 0.6]), _k, _w, 1.0, 100.0, 0.25, _kg, _wg * 0.9, True, True, True, 0, _empty, _empty, _empty, 0, 1.0),
}


@needs_numba
@pytest.mark.parametrize("name", sorted(_PARITY_CASES))
def test_kernel_parity(name):
    """Jitted kernels match the pure-NumPy reference to 1e-10."""
    args = _PARITY_CASES[name]
    plain = K._PLAIN[name](*args)
    jitted = K._JITTED[name](*args)
    if isinstance(plain, tuple):
        for p, j in zip(plain, jitted):
            np.testing.assert_allclose(p, j, rtol=1e-10)
    else:
        np.testing.assert_allclose(plain, jitted, rtol=1e-10)


@needs_numba
def test_finite_diff_matches_np_gradient():
    """The shared finite-difference kernel replicates np.gradient (uniform grid)."""
    dw, d2w = K.finite_diff(_kg, _wg)
    np.testing.assert_allclose(dw, np.gradient(_wg, _kg), rtol=1e-10)
    np.testing.assert_allclose(
        d2w, np.gradient(np.gradient(_wg, _kg), _kg), rtol=1e-10
    )


@needs_numba
def test_toggle_switches_dispatch():
    """use_numba flips which implementation resolve() returns."""
    prev = K.numba_enabled()
    try:
        use_numba(True)
        assert K.resolve("svi_w") is K._JITTED["svi_w"]
        use_numba(False)
        assert K.resolve("svi_w") is K._PLAIN["svi_w"]
    finally:
        use_numba(prev)


def test_use_numba_without_numba_raises(monkeypatch):
    """Enabling the backend without numba installed raises ImportError."""
    monkeypatch.setattr(K, "_NUMBA_AVAILABLE", False)
    monkeypatch.setattr(K, "_enabled", False)
    with pytest.raises(ImportError, match="svi-py\\[numba\\]"):
        use_numba(True)
    use_numba(False)  # disabling is always allowed


@needs_numba
def test_env_var_disables_backend():
    """PYSVI_NUMBA=0 leaves the numba backend off at import."""
    env = dict(os.environ, PYSVI_NUMBA="0")
    out = subprocess.run(
        [sys.executable, "-c",
         "from src.pysvi import _kernels; print(_kernels.numba_enabled())"],
        capture_output=True, text=True, env=env, check=True,
    )
    assert out.stdout.strip() == "False"


@needs_numba
def test_calibration_backend_parity(atm_slice):
    """Both backends converge to the same SVI fit on the same slice."""
    k, w_target, _ = prepare_slice(atm_slice)
    prev = K.numba_enabled()
    try:
        use_numba(False)
        p_np = SVI().calibrate(k, w_target)
        use_numba(True)
        p_nb = SVI().calibrate(k, w_target)
    finally:
        use_numba(prev)
    assert p_np is not None and p_nb is not None
    for name in ("a", "b", "rho", "m", "sigma"):
        np.testing.assert_allclose(p_nb[name], p_np[name], rtol=1e-3, atol=1e-6)


# ── Production architecture: warm_up and per-context backend (#26) ───

def test_warm_up_compiles_and_is_idempotent():
    """warm_up returns elapsed seconds and a second call is near-free."""
    import time
    from src.pysvi import warm_up
    first = warm_up()
    assert first >= 0.0
    t0 = time.perf_counter()
    warm_up()
    assert time.perf_counter() - t0 < max(1.0, first)  # everything cached


def test_backend_context_overrides_global():
    from src.pysvi import backend
    from src.pysvi import _kernels as K
    prev = K.numba_enabled()
    with backend("numpy"):
        assert not K.numba_enabled()
        if K.numba_available():
            with backend("numba"):        # innermost wins
                assert K.numba_enabled()
        assert not K.numba_enabled()
    assert K.numba_enabled() == prev      # global untouched


def test_backend_context_does_not_mutate_global():
    from src.pysvi import backend, use_numba
    from src.pysvi import _kernels as K
    prev = K._enabled
    with backend("numpy"):
        assert K._enabled == prev         # only the contextvar changed
    with pytest.raises(ValueError, match="unknown backend"):
        with backend("fast"):
            pass


def test_backend_context_results_match(atm_slice):
    """Same calibration through both context backends agrees."""
    from src.pysvi import backend
    from src.pysvi.calibration import prepare_slice
    from src.pysvi.models import SVI, numba_available
    k, w, _ = prepare_slice(atm_slice)
    model = SVI()
    with backend("numpy"):
        p_np = model.calibrate(k, w, initialization="multi_start")
    if not numba_available():
        pytest.skip("numba not installed")
    with backend("numba"):
        p_nb = model.calibrate(k, w, initialization="multi_start")
    # fastmath can settle near-tied multi-start basins differently;
    # the fitted CURVES must agree on the quoted range
    w_np = model.total_variance(k, p_np)
    w_nb = model.total_variance(k, p_nb)
    assert float(np.max(np.abs(w_np - w_nb))) < 5e-4


def test_backend_numba_without_numba_raises(monkeypatch):
    from src.pysvi import _kernels as K
    monkeypatch.setattr(K, "_NUMBA_AVAILABLE", False)
    with pytest.raises(ImportError, match="numba is not installed"):
        with K.backend("numba"):
            pass
