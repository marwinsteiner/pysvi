# scripts/bench_numba.py
"""Benchmark: per-slice calibration time, NumPy vs numba backend.

Usage: uv run python scripts/bench_numba.py [n_slices]
"""

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from pysvi import (  # noqa: E402
    ArbitrageFreedom, get_model, calibrate_slice, use_numba, numba_available,
    svi_total_variance,
)


def make_slice(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    F, T = 100.0, 0.25
    k = np.linspace(-0.2, 0.2, 21)
    w = svi_total_variance(k, 0.01, 0.12, -0.6, 0.01, 0.25)
    iv = np.sqrt(w / T) + 0.0002 * rng.standard_normal(k.size)
    return pd.DataFrame({
        "strike": F * np.exp(k), "iv": iv, "maturity": T, "implied_forward": F,
    })


def theta_of(df: pd.DataFrame) -> float:
    return float(np.nanmin(df["iv"] ** 2 * df["maturity"]))


def bench(n_slices: int, arb: ArbitrageFreedom) -> dict:
    slices = [make_slice(s) for s in range(n_slices)]
    T = 0.25
    F = 100.0
    jobs = {
        "svi": ("svi", lambda df: {}),
        "ssvi": ("ssvi", lambda df: {"theta": theta_of(df)}),
        "essvi": ("essvi", lambda df: {"theta": theta_of(df), "theta_ref": theta_of(df)}),
        "jw": ("jw", lambda df: {"T": T}),
        "sabr": ("sabr", lambda df: {"T": T, "F": F, "beta": 1.0}),
        "dsvi": ("dsvi", lambda df: {}),
    }
    out = {}
    for label, (name, kwargs_fn) in jobs.items():
        model = get_model(name, arb)
        calibrate_slice(slices[0], model, **kwargs_fn(slices[0]))  # warmup (JIT)
        t0 = time.perf_counter()
        for df in slices:
            calibrate_slice(df, model, **kwargs_fn(df))
        out[label] = (time.perf_counter() - t0) / n_slices
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    if not numba_available():
        print("numba is not installed; benchmark needs the [numba] extra")
        return
    for arb, arb_label in [
        (ArbitrageFreedom.QUASI, "QUASI"),
        (ArbitrageFreedom.NO_BUTTERFLY, "NO_BUTTERFLY"),
    ]:
        use_numba(False)
        t_np = bench(n, arb)
        use_numba(True)
        t_nb = bench(n, arb)
        print(f"\n{arb_label} ({n} slices, ms per slice)")
        print(f"{'model':<8}{'numpy':>10}{'numba':>10}{'speedup':>10}")
        for m in t_np:
            print(
                f"{m:<8}{t_np[m] * 1e3:>10.2f}{t_nb[m] * 1e3:>10.2f}"
                f"{t_np[m] / t_nb[m]:>9.1f}x"
            )


if __name__ == "__main__":
    main()
