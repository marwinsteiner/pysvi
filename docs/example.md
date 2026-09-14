# From option chain to arbitrage-free surface

One complete, runnable walkthrough: raw call/put quotes with bid/ask spreads, through ingestion and calendar-aware calibration, to a verified surface you can evaluate, price, audit, and persist.

## 1. A realistic synthetic chain

Both legs quoted at every strike, three expiries, bid/ask spreads. Replace this block with your own market data.

```python
import numpy as np
import pandas as pd
from py_vollib.black import black
from pysvi import svi_total_variance

rows = []
r = 0.02
true_params = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}
for T in (0.25, 0.5, 1.0):
    F = 100.0 * np.exp(r * T)
    k = np.linspace(-0.25, 0.25, 21)
    iv = np.sqrt(svi_total_variance(k, **true_params) * (T / 0.25) / T)
    for ki, vi in zip(k, iv):
        K = F * np.exp(ki)
        for flag in ("c", "p"):
            px = black(flag, F, K, T, r, float(vi))
            spread = max(0.02, 0.01 * px)
            rows.append({"strike": K, "expiry": T, "cp": flag,
                         "bid": max(px - spread / 2, 1e-3), "ask": px + spread / 2})
raw = pd.DataFrame(rows)
```

## 2. Ingest

`OptionChain` does the preprocessing: mids, implied forwards from put-call parity, OTM leg selection, Black-76 IV inversion, bid/ask IV bands. Invalid quotes are dropped and counted.

```python
from pysvi import OptionChain

chain = OptionChain.from_dataframe(raw, rate=r, spot=100.0)
chain.panel.head()   # strike, iv, maturity, implied_forward, iv_bid, iv_ask
```

Rates may be a flat float or a term structure `T -> r(T)`; put-call-parity forwards embed dividends automatically, and `spot`/`dividend_yield` only serve as the forward fallback for expiries without put-call pairs.

## 3. Calibrate, calendar-aware

```python
surface = chain.fit(model="svi", enforce_calendar=True,
                    initialization="multi_start")
```

`enforce_calendar` chains each slice into the next slice's NO_CALENDAR penalty automatically; `multi_start` protects against raw SVI's local minima. Any model name works here (`"ssvi"` for a butterfly-free-by-construction surface, `"essvi"` for a jointly fitted term structure, ...).

## 4. Audit

Nothing about the fit is hidden: the report records what went in, what was rejected, how well every slice fitted, and whether the result is arbitrage-free on the quoted range.

```python
print(surface.diagnose())
```

```text
SurfaceFitReport
================
Model:           SVI              Backend:   numba
Objective:       total_variance   Loss:      l2
Initialization:  multi_start      Calendar:  enforced
Slices:          3 ok / 0 failed or rejected
Quotes:          63 in / 63 used

  T        status             quotes   used    iv RMSE   max|res|  k-range
  0.25     ok                     21     21   2.34e-06   5.30e-06  [-0.250, +0.250]
  0.5      ok                     21     21   5.95e-07   1.71e-06  [-0.250, +0.250]
  1        ok                     21     21   4.75e-07   1.04e-06  [-0.250, +0.250]
Fitted 2026-10-11T16:00:00Z | svi-py 1.0.0

Arbitrage diagnostics
=====================
Slice (T=0.25):
  Butterfly arbitrage: none (min g = 2.716e-01 at k = -0.7500)
  Lee wing bounds:     satisfied (left slope = 0.1921, right slope = 0.0482, bound = 2, asymptotic)
Slice (T=0.5):
  Butterfly arbitrage: none (min g = 2.188e-01 at k = -0.7500)
  Lee wing bounds:     satisfied (left slope = 0.3840, right slope = 0.0961, bound = 2, asymptotic)
Slice (T=1):
  Butterfly arbitrage: none (min g = 9.982e-02 at k = -0.7500)
  Lee wing bounds:     satisfied (left slope = 0.7681, right slope = 0.1921, bound = 2, asymptotic)
Calendar arbitrage:    none (min dw = 3.400e-02 at k = 0.1969, between T=0.25 and T=0.5)
Overall:               ARBITRAGE-FREE

Overall: OK
```

`surface.fit_report.ok` is `False` whenever any slice of the input panel failed to calibrate, so partial surfaces cannot pass silently.

## 5. Evaluate and price

```python
surface.iv(100.0, 0.5)          # implied vol at an absolute strike
surface.iv(100.0, 0.7)          # interpolated maturity
surface.atm_vol(0.5), surface.skew(0.5)
surface.price(95.0, 0.7, "put") # Black-76 on the slice forward
surface.delta(95.0, 0.7, "put")
```

## 6. Persist

Calibrating is expensive; evaluating is cheap. Save once, distribute, reload exactly:

```python
surface.save("spx_surface.json")

from pysvi import VolSurface
reloaded = VolSurface.load("spx_surface.json")   # bitwise-identical evaluation
```

The file is versioned JSON (schema_version 1) carrying the model, per-slice parameters and forwards, the interpolation method, and the full fit report, so the provenance travels with the surface.
