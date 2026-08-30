# Quickstart

## Installation

```bash
pip install svi-py
```

Requires Python >= 3.13.

## Calibrating a slice

You need a DataFrame with columns for strike prices, implied volatilities (or raw option prices from which to compute them), time to maturity, and an implied forward price. A typical workflow:

```python
import numpy as np
import pandas as pd
from pysvi import SVI, get_model, calibrate_slice, apply_slice, ArbitrageFreedom

# Suppose df_slice is a single-maturity cross-section with columns:
#   strike, iv, maturity, implied_forward
model = get_model("svi")
params = calibrate_slice(df_slice, model)

# Generate fitted IVs and residuals
fitted = apply_slice(df_slice, params, model)
print(fitted[["strike", "iv", "fitted_iv", "residual_iv"]])
```

## Model-specific arguments

Some parametrizations need extra inputs per slice.

For SSVI/eSSVI, pass the ATM total variance:

```python
model = get_model("ssvi")
theta = float(np.nanmin(df_slice["iv"] ** 2 * df_slice["maturity"]))
params = calibrate_slice(df_slice, model, theta=theta)
```

For jump-wings, pass the time to expiry:

```python
model = get_model("jw")
T = float(df_slice["maturity"].iloc[0])
params = calibrate_slice(df_slice, model, T=T)
```

DirectSVI requires no extra arguments — it fits directly from log-moneyness and total variance:

```python
model = get_model("dsvi")
params = calibrate_slice(df_slice, model)
```

For SABR, pass the time to expiry and the forward, and optionally fix the CEV exponent β (default 0.5):

```python
model = get_model("sabr")
T = float(df_slice["maturity"].iloc[0])
F = float(df_slice["implied_forward"].iloc[0])
params = calibrate_slice(df_slice, model, T=T, F=F, beta=1.0)  # beta=1 for FX/equity
```

See {doc}`models/index` for the full catalogue and when to use which model.

## Numba acceleration

Calibration hot paths have JIT-compiled twins. Install the optional extra:

```bash
pip install "svi-py[numba]"
```

When numba is installed the accelerated backend is on automatically — no code
changes needed. Toggle it at runtime with `pysvi.use_numba(False)` /
`pysvi.use_numba(True)`, or disable it at import time with the environment
variable `PYSVI_NUMBA=0`. Both backends produce the same results to within
floating-point rounding.

Expect roughly 2-6x faster calibration when grid-based arbitrage penalties
are active (`NO_BUTTERFLY` / `NO_CALENDAR`), with the largest gains for SABR;
unconstrained (`QUASI`) fits on small slices are dominated by optimizer
overhead, so gains there are modest. Kernels compile on first use in each
process (a few seconds); measure your own workload with
`python scripts/bench_numba.py`.

## Where do the inputs come from?

`svi-py` expects you to already have implied volatilities and forward prices. If you're starting from raw option prices, the library provides helpers:

- `compute_ivs_vectorized` computes Black-Scholes-Merton implied vols from option mid-prices via `py_vollib`.
- `calculate_implied_forward` estimates the forward price from put-call parity:

$$F = K + e^{rT}(C - P)$$

- `choose_leg` selects the OTM leg (calls for $K \geq F$, puts for $K < F$) for cleaner vol quotes.

You need a panel of **contemporaneous call and put option prices** across multiple strikes for at least one maturity. The richer the strike grid, the better the calibration. See {doc}`calibration` for pipeline details.
