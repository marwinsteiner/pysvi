"""Calibration controls on real data: objectives, losses, starts, numba.

Every iterative model accepts four orthogonal controls through
``calibrate`` / ``calibrate_slice`` / the surface constructors:

* ``objective`` -- which residual the optimizer sees:
  ``total_variance`` (default), ``implied_vol``, ``price`` (Black
  call), ``vega_weighted``, ``bid_ask`` (needs ``w_bid``/``w_ask``)
* ``loss`` -- how residuals aggregate: ``l2`` (default), ``huber``,
  ``soft_l1``, ``cauchy`` (scipy least_squares convention)
* ``f_scale`` -- robust-loss scale; defaults to 1.4826 * MAD of the
  residuals at a pilot l2 fit, override with any float
* ``initialization`` -- ``default``, ``jump_wings`` (SVI/NaturalSVI:
  data-driven wing readoff), ``multi_start`` (deterministic 16-start
  grid, best converged fit wins)

Plus the arbitrage penalty flags (``ArbitrageFreedom``) and the numba
backend toggles. All demonstrated below on one real SPY expiry.

Usage::

    uv run examples/04_calibration_controls.py
"""

import time

import numpy as np

from pysvi import (
    ArbitrageFreedom, get_model, numba_available, prepare_slice, use_numba,
)

from _snapshot import load_snapshot
from importlib import import_module

build_slice = import_module("03_single_slice_models").build_slice


def rmse_iv(model, params, k, w, T) -> float:
    w_fit = model.total_variance(k, params)
    return float(np.sqrt(np.mean(
        (np.sqrt(np.maximum(w_fit, 0) / T) - np.sqrt(np.maximum(w, 0) / T)) ** 2
    )))


def main() -> None:
    df, meta = load_snapshot()
    df_slice = build_slice(df, meta)
    T = float(df_slice["maturity"].iloc[0])
    k, w, F = prepare_slice(df_slice)
    print(f"slice: T={T:.3f}y, {len(k)} quotes\n")

    model = get_model("svi", ArbitrageFreedom.NO_BUTTERFLY)

    # ── Objectives ───────────────────────────────────────────────────
    # Same data, different residual spaces. price/vega_weighted need T
    # (and price discounts nothing -- it works in forward terms).
    for objective in ("total_variance", "implied_vol", "price", "vega_weighted"):
        params = model.calibrate(
            k, w, objective=objective, T=T, initialization="multi_start",
        )
        print(f"objective={objective:<15} IV RMSE "
              f"{rmse_iv(model, params, k, w, T) * 1e4:6.1f} bp")

    # The bid_ask objective fits inside the quoted band instead of to
    # the mid: residuals are zero anywhere between w_bid and w_ask.
    # Real bid/ask IVs come from the snapshot quotes (example 05 gets
    # them for free from OptionChain's iv_bid/iv_ask columns).
    spread_w = 0.04 * w                      # stand-in half-spread band
    params_ba = model.calibrate(
        k, w,
        objective="bid_ask",
        w_bid=w - spread_w,
        w_ask=w + spread_w,
        initialization="multi_start",
    )
    print(f"objective=bid_ask         IV RMSE "
          f"{rmse_iv(model, params_ba, k, w, T) * 1e4:6.1f} bp "
          f"(any fit inside the band is 'perfect')\n")

    # ── Robust losses and f_scale ────────────────────────────────────
    # Corrupt one wing quote by 10 vol points: l2 chases the outlier,
    # robust losses cap its influence. f_scale sets where "outlier"
    # begins -- the pilot-fit MAD default is usually right; override it
    # to trade robustness against efficiency.
    w_dirty = w.copy()
    iv_dirty = np.sqrt(w_dirty[0] / T) + 0.10
    w_dirty[0] = iv_dirty ** 2 * T
    clean_mask = np.arange(len(k)) != 0
    for loss in ("l2", "huber", "soft_l1", "cauchy"):
        params = model.calibrate(
            k, w_dirty, loss=loss, initialization="multi_start",
        )
        err = rmse_iv(model, params, k[clean_mask], w[clean_mask], T)
        print(f"loss={loss:<8} (auto f_scale)   clean-quote RMSE {err * 1e4:6.1f} bp")
    params = model.calibrate(
        k, w_dirty, loss="cauchy", f_scale=0.001, initialization="multi_start",
    )
    err = rmse_iv(model, params, k[clean_mask], w[clean_mask], T)
    print(f"loss=cauchy   f_scale=0.001    clean-quote RMSE {err * 1e4:6.1f} bp\n")

    # ── Initializations ──────────────────────────────────────────────
    # Raw SVI's landscape has a genuine bad basin reachable from the
    # default start on some platforms; multi_start is the reliable
    # escape and what the docs recommend for production fits.
    for init in ("default", "jump_wings", "multi_start"):
        t0 = time.perf_counter()
        params = model.calibrate(k, w, initialization=init)
        ms = (time.perf_counter() - t0) * 1e3
        print(f"initialization={init:<12} IV RMSE "
              f"{rmse_iv(model, params, k, w, T) * 1e4:6.1f} bp   {ms:6.1f} ms")

    # ── Arbitrage penalty flags ──────────────────────────────────────
    # QUASI = parameter-bound constraints only; NO_BUTTERFLY adds a
    # density penalty grid; NO_CALENDAR penalizes crossing a previous
    # slice (pass w_prev -- calibrate_surface chains it automatically,
    # see docs/arbitrage for the manual protocol).
    combined = ArbitrageFreedom.NO_BUTTERFLY | ArbitrageFreedom.NO_CALENDAR
    print(f"\nArbitrageFreedom flags compose: {combined!r}")

    # ── Numba backend ────────────────────────────────────────────────
    # With svi-py[numba] installed the JIT backend is on automatically;
    # use_numba(False) switches to pure NumPy at runtime, and the
    # environment variable PYSVI_NUMBA=0 disables it at import time.
    print(f"numba_available() = {numba_available()}")
    if numba_available():
        for enabled in (True, False):
            use_numba(enabled)
            t0 = time.perf_counter()
            model.calibrate(k, w, initialization="multi_start")
            ms = (time.perf_counter() - t0) * 1e3
            print(f"use_numba({enabled!s:<5}) multi_start fit: {ms:6.1f} ms")
        use_numba(True)

    # Production services: use_numba mutates process-global state, which
    # races under concurrency. backend() pins the choice per context
    # (thread/async-task local), and warm_up() moves the one-off JIT
    # compilation cost (~15 s for every kernel of every model) to
    # service startup instead of the first live request.
    from pysvi import backend, warm_up
    secs = warm_up()
    print(f"warm_up(): all kernels compiled in {secs:.1f} s "
          "(near-zero when already warm)")
    with backend("numpy"):
        t0 = time.perf_counter()
        model.calibrate(k, w, initialization="multi_start")
        print(f"backend('numpy') fit:  {(time.perf_counter() - t0) * 1e3:6.1f} ms")
    if numba_available():
        with backend("numba"):
            t0 = time.perf_counter()
            model.calibrate(k, w, initialization="multi_start")
            print(f"backend('numba') fit:  {(time.perf_counter() - t0) * 1e3:6.1f} ms")


if __name__ == "__main__":
    main()
