# src/pysvi/_kernels.py
"""Numerical kernels with optional numba acceleration.

Every kernel is written once, in numba-compatible NumPy, and compiled to a
jitted twin at import time when numba is installed. The active implementation
is looked up at call time through :func:`resolve`, so the backend can be
toggled at runtime without re-importing.

Composite functions (jw_total_variance and the fused per-model calibration
objectives) are built by factories closing over their leaf kernels, so the
same source produces a pure-NumPy version (closing over plain functions) and
a jitted version (closing over jitted ones).

Kernels compile once per process on first use (a few seconds). Disk caching
(``cache=True``) is deliberately not used: numba's cache pickles the module
environment by module name, and the same source imported under two names
(``pysvi`` installed vs ``src.pysvi`` in this repo's tests) crashes with
ModuleNotFoundError when one's cache entry is loaded by the other.

All jitted kernels use ``fastmath=True``: results may differ from the NumPy
backend at the level of floating-point rounding (~1e-15 relative), far below
calibration noise.
"""

import os

import numpy as np

try:
    import numba as _nb

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _nb = None
    _NUMBA_AVAILABLE = False

_env = os.environ.get("PYSVI_NUMBA", "").strip().lower()
_enabled = _NUMBA_AVAILABLE and _env not in ("0", "false", "off")


def numba_available() -> bool:
    """True if numba is importable in this environment."""
    return _NUMBA_AVAILABLE


def numba_enabled() -> bool:
    """True if jitted kernels are currently active."""
    return _enabled


def use_numba(enabled: bool = True) -> None:
    """Enable or disable the numba backend at runtime.

    Raises ImportError if numba is requested but not installed.
    """
    global _enabled
    if enabled and not _NUMBA_AVAILABLE:
        raise ImportError(
            "numba is not installed; install the extra with "
            '`pip install "svi-py[numba]"` to enable accelerated kernels.'
        )
    _enabled = bool(enabled)


# ── Leaf kernels ─────────────────────────────────────────────────────
# Plain NumPy definitions; jitted twins are created from the same source
# at the bottom of this module.


def svi_w(k, a, b, rho, m, sigma):
    z = k - m
    return a + b * (rho * z + np.sqrt(z * z + sigma * sigma))


def ssvi_w(k, theta, rho, phi):
    term1 = 1.0 + rho * phi * k
    term2 = np.sqrt((phi * k + rho) ** 2 + (1.0 - rho**2))
    return 0.5 * theta * (term1 + term2)


def essvi_w(k, theta, rho_theta, phi):
    inside = (phi * k + rho_theta) ** 2 + (1.0 - rho_theta**2)
    term1 = 1.0 + rho_theta * phi * k
    term2 = np.sqrt(inside)
    return 0.5 * theta * (term1 + term2)


def sabr_vol(k, alpha, beta, rho, nu, F, T):
    one_m_beta = 1.0 - beta
    L = -k  # log(F/K) = -log(K/F)
    fk_pow = F**one_m_beta * np.exp(0.5 * one_m_beta * k)
    denom_series = (
        1.0 + one_m_beta**2 / 24.0 * L**2 + one_m_beta**4 / 1920.0 * L**4
    )
    z = (nu / alpha) * fk_pow * L
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))
    # Where |z| is tiny x_z is too; substitute 1.0 to keep the (discarded)
    # division defined without errstate, which numba does not support.
    x_z_safe = np.where(np.abs(x_z) < 1e-300, 1.0, x_z)
    z_over_x = np.where(np.abs(z) < 1e-8, 1.0 - 0.5 * rho * z, z / x_z_safe)
    bracket = 1.0 + (
        one_m_beta**2 * alpha**2 / (24.0 * fk_pow**2)
        + rho * beta * nu * alpha / (4.0 * fk_pow)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    ) * T
    return alpha / (fk_pow * denom_series) * z_over_x * bracket


def directsvi_w(k, z0, z1, z2, z3, z4, z5):
    A = z1
    B = z2 * k + z4
    C = z0 * k**2 + z3 * k + z5
    discriminant = B**2 - 4.0 * A * C
    discriminant = np.maximum(discriminant, 0.0)
    y1 = (-B + np.sqrt(discriminant)) / (2.0 * A)
    y2 = (-B - np.sqrt(discriminant)) / (2.0 * A)
    y = np.where(y1 >= 0, y1, y2)
    return np.maximum(y, 0.0)


def svi_derivs(k, a, b, rho, m, sigma):
    z = k - m
    r = np.sqrt(z * z + sigma * sigma)
    w = a + b * (rho * z + r)
    dw = b * (rho + z / r)
    d2w = b * sigma**2 / r**3
    return w, dw, d2w


def ssvi_derivs(k, theta, rho, phi):
    u = phi * k + rho
    disc = np.sqrt(u**2 + 1.0 - rho**2)
    w = 0.5 * theta * (1.0 + rho * phi * k + disc)
    dw = 0.5 * theta * phi * (rho + u / disc)
    d2w = 0.5 * theta * phi**2 * (1.0 - rho**2) / disc**3
    return w, dw, d2w


def butterfly_penalty(k, w, dw, d2w):
    g = (1.0 - k * dw / (2.0 * w)) ** 2 - (dw**2) / 4.0 * (1.0 / w + 0.25) + d2w / 2.0
    violations = np.maximum(-g, 0.0)
    return np.sum(violations**2)


def calendar_penalty(w_current, w_prev):
    diff = w_prev - w_current  # positive where calendar arb exists
    violations = np.maximum(diff, 0.0)
    return np.sum(violations**2)


def finite_diff(x, w):
    """First and second derivative on a uniform grid.

    Central differences in the interior, first-order one-sided at the
    edges — equivalent to two passes of ``np.gradient`` (edge_order=1)
    for uniformly spaced ``x``. Shared by both backends (``np.gradient``
    itself is not numba-supported).
    """
    n = w.shape[0]
    dw = np.empty(n)
    for i in range(1, n - 1):
        dw[i] = (w[i + 1] - w[i - 1]) / (x[i + 1] - x[i - 1])
    dw[0] = (w[1] - w[0]) / (x[1] - x[0])
    dw[n - 1] = (w[n - 1] - w[n - 2]) / (x[n - 1] - x[n - 2])
    d2w = np.empty(n)
    for i in range(1, n - 1):
        d2w[i] = (dw[i + 1] - dw[i - 1]) / (x[i + 1] - x[i - 1])
    d2w[0] = (dw[1] - dw[0]) / (x[1] - x[0])
    d2w[n - 1] = (dw[n - 1] - dw[n - 2]) / (x[n - 1] - x[n - 2])
    return dw, d2w


# ── Composite factories ──────────────────────────────────────────────
# Each factory closes over its leaf kernels so the identical source can be
# instantiated against plain or jitted leaves.


def make_jw_w(svi_w_fn):
    def jw_w(k, v_t, psi_t, p_t, c_t, v_tilde_t, T):
        b = (p_t + c_t) / 2.0
        if b < 1e-12:
            return np.full_like(k, v_t * T)
        rho = 1.0 - p_t / b
        beta = rho - 2.0 * psi_t * np.sqrt(T) / b
        if beta > 0.9999:
            beta = 0.9999
        elif beta < -0.9999:
            beta = -0.9999
        if abs(beta) < 1e-12:
            alpha = 0.0
        else:
            sign_beta = 1.0 if beta > 0.0 else -1.0
            alpha = sign_beta * np.sqrt(max(1.0 / (beta * beta) - 1.0, 0.0))
        sign_alpha = 1.0 if alpha > 0.0 else (-1.0 if alpha < 0.0 else 0.0)
        denom = (
            -rho
            + sign_alpha * np.sqrt(1.0 + alpha * alpha)
            - alpha * np.sqrt(1.0 - rho * rho)
        )
        if abs(denom) < 1e-12:
            m = 0.0
        else:
            m = (v_t - v_tilde_t) * T / (b * denom)
        sigma = max(abs(alpha * m), 1e-12)
        a = v_tilde_t * T - b * sigma * np.sqrt(1.0 - rho * rho)
        return svi_w_fn(k, a, b, rho, m, sigma)

    return jw_w


def make_svi_objective(svi_w_fn, svi_derivs_fn, butterfly_fn, calendar_fn):
    def objective(p, k, w_target, k_grid, w_prev, check_bf, check_cal, has_prev):
        a, b, rho, m, sigma = p[0], p[1], p[2], p[3], p[4]
        penalty = 0.0
        if b <= 0.0:
            penalty += 1e6 * (1.0 - b) ** 2
        if abs(rho) >= 0.999:
            penalty += 1e6 * (abs(rho) - 0.999) ** 2
        if sigma <= 0.0:
            penalty += 1e6 * (1.0 - sigma) ** 2
        w_model = svi_w_fn(k, a, b, rho, m, sigma)
        mse = np.mean((w_target - w_model) ** 2)
        if (check_bf or check_cal) and b > 0.0 and sigma > 0.0:
            w_g = svi_w_fn(k_grid, a, b, rho, m, sigma)
            if check_bf:
                _, dw_g, d2w_g = svi_derivs_fn(k_grid, a, b, rho, m, sigma)
                penalty += 1e4 * butterfly_fn(k_grid, w_g, dw_g, d2w_g)
            if check_cal and has_prev:
                penalty += 1e4 * calendar_fn(w_g, w_prev)
        return mse + penalty

    return objective


def make_ssvi_objective(ssvi_w_fn, ssvi_derivs_fn, butterfly_fn, calendar_fn):
    def objective(
        p, k, w_target, theta, k_grid, w_prev, check_bf, check_cal, has_prev
    ):
        rho, eta = p[0], p[1]
        penalty = 0.0
        if abs(rho) >= 0.999:
            penalty += 1e6 * (abs(rho) - 0.999) ** 2
        if eta <= 0.0:
            penalty += 1e6 * (1.0 - eta) ** 2
        phi = eta / np.sqrt(theta)
        w_model = ssvi_w_fn(k, theta, rho, phi)
        mse = np.mean((w_target - w_model) ** 2)
        if (check_bf or check_cal) and eta > 0.0:
            w_g = ssvi_w_fn(k_grid, theta, rho, phi)
            if check_bf:
                _, dw_g, d2w_g = ssvi_derivs_fn(k_grid, theta, rho, phi)
                penalty += 1e4 * butterfly_fn(k_grid, w_g, dw_g, d2w_g)
            if check_cal and has_prev:
                penalty += 1e4 * calendar_fn(w_g, w_prev)
        return mse + penalty

    return objective


def make_essvi_objective(essvi_w_fn, ssvi_derivs_fn, butterfly_fn, calendar_fn):
    def objective(
        p, k, w_target, theta, theta_ref, k_grid, w_prev,
        check_bf, check_cal, has_prev,
    ):
        rho0, rho1, alpha, eta = p[0], p[1], p[2], p[3]
        penalty = 0.0
        if eta <= 0.0:
            penalty += 1e6 * (1.0 - eta) ** 2
        theta_ratio = theta / max(theta_ref, 1e-12)
        rho_theta = rho0 + rho1 * theta_ratio**alpha
        if rho_theta > 0.999:
            rho_theta = 0.999
        elif rho_theta < -0.999:
            rho_theta = -0.999
        phi = eta / np.sqrt(theta)
        w_model = essvi_w_fn(k, theta, rho_theta, phi)
        mse = np.mean((w_target - w_model) ** 2)
        penalty += 1e2 * max(0.0, abs(rho_theta) - 0.95)
        if (check_bf or check_cal) and eta > 0.0:
            w_g = essvi_w_fn(k_grid, theta, rho_theta, phi)
            if check_bf:
                _, dw_g, d2w_g = ssvi_derivs_fn(k_grid, theta, rho_theta, phi)
                penalty += 1e4 * butterfly_fn(k_grid, w_g, dw_g, d2w_g)
            if check_cal and has_prev:
                penalty += 1e4 * calendar_fn(w_g, w_prev)
        return mse + penalty

    return objective


def make_jw_objective(jw_w_fn, svi_derivs_fn, butterfly_fn, calendar_fn):
    def objective(
        p, k, w_target, T, k_grid, w_prev, check_bf, check_cal, has_prev
    ):
        v_t, psi_t, p_t, c_t, v_tilde_t = p[0], p[1], p[2], p[3], p[4]
        penalty = 0.0
        if p_t < 0.0:
            penalty += 1e6 * p_t**2
        if c_t < 0.0:
            penalty += 1e6 * c_t**2
        if v_tilde_t <= 0.0:
            penalty += 1e6 * (1.0 - v_tilde_t) ** 2
        if v_t <= 0.0:
            penalty += 1e6 * (1.0 - v_t) ** 2
        if v_tilde_t > v_t:
            penalty += 1e4 * (v_tilde_t - v_t) ** 2
        w_model = jw_w_fn(k, v_t, psi_t, p_t, c_t, v_tilde_t, T)
        mse = np.mean((w_target - w_model) ** 2)
        if (
            (check_bf or check_cal)
            and v_t > 0.0
            and v_tilde_t > 0.0
            and p_t >= 0.0
            and c_t >= 0.0
        ):
            w_g = jw_w_fn(k_grid, v_t, psi_t, p_t, c_t, v_tilde_t, T)
            if check_bf:
                b = (p_t + c_t) / 2.0
                if b > 1e-12:
                    rho = 1.0 - p_t / b
                    beta = rho - 2.0 * psi_t * np.sqrt(T) / b
                    if beta > 0.9999:
                        beta = 0.9999
                    elif beta < -0.9999:
                        beta = -0.9999
                    sign_beta = 1.0 if beta > 0.0 else (-1.0 if beta < 0.0 else 0.0)
                    alpha_jw = sign_beta * np.sqrt(
                        max(1.0 / (beta * beta) - 1.0, 0.0)
                    )
                    sign_alpha = (
                        1.0 if alpha_jw > 0.0 else (-1.0 if alpha_jw < 0.0 else 0.0)
                    )
                    denom = (
                        -rho
                        + sign_alpha * np.sqrt(1.0 + alpha_jw**2)
                        - alpha_jw * np.sqrt(1.0 - rho**2)
                    )
                    if abs(denom) > 1e-12:
                        m = (v_t - v_tilde_t) * T / (b * denom)
                    else:
                        m = 0.0
                    sigma = max(abs(alpha_jw * m), 1e-12)
                    a = v_tilde_t * T - b * sigma * np.sqrt(1.0 - rho**2)
                    _, dw_g, d2w_g = svi_derivs_fn(k_grid, a, b, rho, m, sigma)
                    penalty += 1e4 * butterfly_fn(k_grid, w_g, dw_g, d2w_g)
            if check_cal and has_prev:
                penalty += 1e4 * calendar_fn(w_g, w_prev)
        return mse + penalty

    return objective


def make_sabr_objective(sabr_vol_fn, finite_diff_fn, butterfly_fn, calendar_fn):
    def objective(
        p, k, w_target, beta, F, T, k_grid, w_prev,
        check_bf, check_cal, has_prev,
    ):
        alpha, rho, nu = p[0], p[1], p[2]
        penalty = 0.0
        if alpha <= 0.0:
            penalty += 1e6 * (1.0 - alpha) ** 2
            return 1e6 + penalty
        if abs(rho) >= 0.999:
            penalty += 1e6 * (abs(rho) - 0.999) ** 2
            if rho > 0.998:
                rho = 0.998
            elif rho < -0.998:
                rho = -0.998
        if nu < 0.0:
            penalty += 1e6 * nu**2
            nu = 0.0
        sig = sabr_vol_fn(k, alpha, beta, rho, nu, F, T)
        w_model = sig * sig * T
        mse = np.mean((w_target - w_model) ** 2)
        if check_bf or check_cal:
            sig_g = sabr_vol_fn(k_grid, alpha, beta, rho, nu, F, T)
            w_g = sig_g * sig_g * T
            if check_bf:
                dw_g, d2w_g = finite_diff_fn(k_grid, w_g)
                penalty += 1e4 * butterfly_fn(k_grid, w_g, dw_g, d2w_g)
            if check_cal and has_prev:
                penalty += 1e4 * calendar_fn(w_g, w_prev)
        return mse + penalty

    return objective


# ── Backend registry ─────────────────────────────────────────────────

jw_w = make_jw_w(svi_w)

_PLAIN = {
    "svi_w": svi_w,
    "ssvi_w": ssvi_w,
    "essvi_w": essvi_w,
    "jw_w": jw_w,
    "sabr_vol": sabr_vol,
    "directsvi_w": directsvi_w,
    "svi_derivs": svi_derivs,
    "ssvi_derivs": ssvi_derivs,
    "butterfly": butterfly_penalty,
    "calendar": calendar_penalty,
    "finite_diff": finite_diff,
    "svi_obj": make_svi_objective(svi_w, svi_derivs, butterfly_penalty, calendar_penalty),
    "ssvi_obj": make_ssvi_objective(ssvi_w, ssvi_derivs, butterfly_penalty, calendar_penalty),
    "essvi_obj": make_essvi_objective(essvi_w, ssvi_derivs, butterfly_penalty, calendar_penalty),
    "jw_obj": make_jw_objective(jw_w, svi_derivs, butterfly_penalty, calendar_penalty),
    "sabr_obj": make_sabr_objective(sabr_vol, finite_diff, butterfly_penalty, calendar_penalty),
}

_JITTED: dict = {}
if _NUMBA_AVAILABLE:
    _jit = _nb.njit(fastmath=True)

    _svi_w_nb = _jit(svi_w)
    _ssvi_w_nb = _jit(ssvi_w)
    _essvi_w_nb = _jit(essvi_w)
    _sabr_vol_nb = _jit(sabr_vol)
    _directsvi_w_nb = _jit(directsvi_w)
    _svi_derivs_nb = _jit(svi_derivs)
    _ssvi_derivs_nb = _jit(ssvi_derivs)
    _butterfly_nb = _jit(butterfly_penalty)
    _calendar_nb = _jit(calendar_penalty)
    _finite_diff_nb = _jit(finite_diff)
    _jw_w_nb = _jit(make_jw_w(_svi_w_nb))

    _JITTED = {
        "svi_w": _svi_w_nb,
        "ssvi_w": _ssvi_w_nb,
        "essvi_w": _essvi_w_nb,
        "jw_w": _jw_w_nb,
        "sabr_vol": _sabr_vol_nb,
        "directsvi_w": _directsvi_w_nb,
        "svi_derivs": _svi_derivs_nb,
        "ssvi_derivs": _ssvi_derivs_nb,
        "butterfly": _butterfly_nb,
        "calendar": _calendar_nb,
        "finite_diff": _finite_diff_nb,
        "svi_obj": _jit(make_svi_objective(_svi_w_nb, _svi_derivs_nb, _butterfly_nb, _calendar_nb)),
        "ssvi_obj": _jit(make_ssvi_objective(_ssvi_w_nb, _ssvi_derivs_nb, _butterfly_nb, _calendar_nb)),
        "essvi_obj": _jit(make_essvi_objective(_essvi_w_nb, _ssvi_derivs_nb, _butterfly_nb, _calendar_nb)),
        "jw_obj": _jit(make_jw_objective(_jw_w_nb, _svi_derivs_nb, _butterfly_nb, _calendar_nb)),
        "sabr_obj": _jit(make_sabr_objective(_sabr_vol_nb, _finite_diff_nb, _butterfly_nb, _calendar_nb)),
    }


def resolve(name: str):
    """Return the active implementation (jitted if enabled) for a kernel."""
    if _enabled and name in _JITTED:
        return _JITTED[name]
    return _PLAIN[name]
