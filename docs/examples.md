# Real-data examples

The repository ships a top-level [`examples/`](https://github.com/marwinsteiner/pysvi/tree/main/examples)
directory: five runnable scripts that take a **real SPY option chain**
(fetched via [yfinance](https://github.com/ranaroussi/yfinance)) from
raw quotes to a priced, verified, serialized surface. Together they
exercise the complete public API — every endpoint and every parameter —
on one shared market snapshot, so they double as living documentation
of the library surface.

| Script | Covers |
|---|---|
| [`01_fetch_chain_yfinance.py`](https://github.com/marwinsteiner/pysvi/blob/main/examples/01_fetch_chain_yfinance.py) | Snapshot discipline: a contemporaneous chain with **no lookahead**; a four-pillar Treasury curve fitted with [interest-rate-models](https://pypi.org/project/interest-rate-models/); `parse_ticker_info` |
| [`02_implied_vol_and_forwards.py`](https://github.com/marwinsteiner/pysvi/blob/main/examples/02_implied_vol_and_forwards.py) | `calculate_implied_forward` (flat and term-structure rates), `choose_leg`, `compute_ivs_vectorized`, `prepare_slice`; implied-vol inversion methods with sources |
| [`03_single_slice_models.py`](https://github.com/marwinsteiner/pysvi/blob/main/examples/03_single_slice_models.py) | All seven parametrizations, `apply_slice`, the module-level total-variance functions, the raw/natural bijection, derivatives/density/wing slopes, `check_slice_arbitrage` |
| [`04_calibration_controls.py`](https://github.com/marwinsteiner/pysvi/blob/main/examples/04_calibration_controls.py) | Every `objective`, `loss`, `f_scale`, `initialization`; `ArbitrageFreedom` flags; the numba toggles |
| [`05_surface_pipeline.py`](https://github.com/marwinsteiner/pysvi/blob/main/examples/05_surface_pipeline.py) | `OptionChain`, `calibrate_surface`, `VolSurface` evaluation and interpolation, `diagnose`, Black-76 Greeks, `save`/`load` |

Only script 01 needs the network; a committed sample snapshot lets
02–05 run immediately:

```bash
uv run --with yfinance --with interest-rate-models     examples/01_fetch_chain_yfinance.py                      # optional refresh
uv run examples/02_implied_vol_and_forwards.py               # ... through 05
```

Rates are a real curve, not a constant: script 01 fits a four-pillar
Treasury zero curve (13w/5y/10y/30y) with
[interest-rate-models](https://pypi.org/project/interest-rate-models/)
(`import interest_rate_models as irm`; `irm.DiscountCurve`) and
persists it densely in the snapshot metadata. Neither yfinance nor
interest-rate-models is a dependency of `svi-py` -- the calibration
API takes a plain callable `T -> r(T)`, and the offline scripts
reconstruct it from the file with numpy alone.

## No lookahead

A volatility surface is a picture of the market at one instant; mixing
quotes observed at different times produces phantom arbitrage and
unstable fits. The examples enforce contemporaneity mechanically:
script 01 records a single snapshot timestamp, fetches quotes, spot,
and rates in one pass, filters staleness using only information
available at that timestamp, and persists everything to disk. The
other scripts read the files and compute time-to-expiry against the
recorded timestamp — never the wall clock. Apply the same discipline
to production calibration: only data observable at the valuation time.

## Implied-vol inversion methods

There is no closed-form inverse of the Black-Scholes price, so every
implied vol is the output of an inversion algorithm:

- **Black-76 root-finding** — the textbook route (Newton/Brent on the
  pricing formula); fragile near intrinsic value.
- **"Let's Be Rational"** (Peter Jäckel, *Wilmott* 2015) — the de facto
  standard: rational approximations plus at most two Householder steps
  reach full machine precision at roughly the cost of two price
  evaluations. [Paper](http://www.jaeckel.org/LetsBeRational.pdf)
  ([journal](https://doi.org/10.1002/wilm.10395)),
  [reference implementation](https://github.com/vollib/lets_be_rational).
  **This is what `svi-py` uses internally**: `compute_ivs_vectorized`
  and `OptionChain` invert through `py_vollib`, whose engine is
  `py_lets_be_rational`.
- **Volfi** (Wolfgang Schadner — also the author of the DirectSVI
  parametrization shipped in this library) — newer still: an
  *explicit*, non-iterative inverse via a generalized-inverse-Gaussian
  quantile representation with an optional single Halley refinement.
  [Paper](https://arxiv.org/abs/2604.24480),
  [code](https://github.com/wol-fi/volfi). See also
  [fast-vollib](https://arxiv.org/abs/2604.27210) for vectorized LBR
  at scale.

## Maintenance policy

The examples are part of the API contract: whenever the public API
changes, `examples/` and the documentation change in the same pull
request. Drift between the examples and the library is a bug.
