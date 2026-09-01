# svi-py

Stochastic volatility inspired (SVI) parametrizations of the implied volatility surface in Python — plus the SABR stochastic volatility model.

Given a panel of contemporaneous European call and put option prices across strikes and maturities, `svi-py` calibrates smooth, arbitrage-aware total variance surfaces. It handles the full pipeline: implied vol extraction, forward estimation via put-call parity, OTM leg selection, and per-slice calibration with configurable no-arbitrage constraints.

## Features

- **Seven parametrizations** behind one interface: raw SVI, natural SVI, SSVI, eSSVI, jump-wings, DirectSVI, and SABR
- **Configurable no-arbitrage enforcement**: butterfly (density) and calendar-spread penalties, combinable as flags
- **Full data pipeline**: BSM implied vols from prices, implied forwards from put-call parity, OTM leg selection, slice preparation
- **Robust calibration**: L-BFGS-B with automatic Nelder-Mead fallback; closed-form fitting for DirectSVI
- **Optional numba acceleration**: JIT-compiled kernels behind a runtime toggle (`pip install "svi-py[numba]"`), 2-6x faster arbitrage-constrained calibration
- **Fitted surface object**: `VolSurface.fit(df)` gives evaluation (IVs, ATM level/skew/curvature), arbitrage verification, and Black-76 pricing and Greeks in one object

## Installation

```bash
pip install svi-py
```

Requires Python >= 3.13.

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
surface
models/index
arbitrage
calibration
api
contributing
```
