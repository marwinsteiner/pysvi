# Examples

Runnable, end-to-end examples on **real market data** (SPY options via
[yfinance](https://github.com/ranaroussi/yfinance)). Together the five
scripts exercise the complete public API -- every endpoint and every
parameter -- on the same snapshot, so they double as living
documentation of the library surface.

| Script | What it covers |
|---|---|
| `01_fetch_chain_yfinance.py` | Snapshot discipline: fetching a contemporaneous chain **without lookahead**; a real four-pillar Treasury curve (13w/5y/10y/30y) fitted with [interest-rate-models](https://pypi.org/project/interest-rate-models/) (`irm.DiscountCurve`, plus a Vasicek fit); `parse_ticker_info` |
| `02_implied_vol_and_forwards.py` | `calculate_implied_forward` (flat + term-structure rates), `choose_leg`, `compute_ivs_vectorized`, `prepare_slice`; implied-vol inversion methods (Black-76, LBR, Volfi) with sources |
| `03_single_slice_models.py` | All 7 parametrizations via `get_model`/`calibrate_slice`/`apply_slice`, the module-level `*_total_variance` functions, `natural_to_raw`/`raw_to_natural`, `derivatives`/`dw_dk`/`d2w_dk2`/`density`/`wing_slopes`/`fd_step`, `check_slice_arbitrage` |
| `04_calibration_controls.py` | Every `objective` (incl. `bid_ask`), every `loss`, `f_scale`, every `initialization`, `ArbitrageFreedom` flags, `use_numba`/`numba_available`/`PYSVI_NUMBA` |
| `05_surface_pipeline.py` | `OptionChain.from_dataframe` (all parameters), `chain.fit`, `calibrate_surface`, `VolSurface.fit` + direct construction, all evaluation methods, maturity interpolation (`total_variance`/`theta`), `diagnose`/`check_arbitrage`, all Black-76 Greeks, `save`/`load` |

## Running

Only script 01 needs the network (and the `yfinance` extra):

```bash
uv run --with yfinance --with interest-rate-models examples/01_fetch_chain_yfinance.py   # fetch snapshot
uv run examples/02_implied_vol_and_forwards.py
uv run examples/03_single_slice_models.py
uv run examples/04_calibration_controls.py
uv run examples/05_surface_pipeline.py
```

(Outside this repo: `pip install svi-py yfinance interest-rate-models`
and run with plain `python`.) A committed sample snapshot in
`examples/data/` lets 02-05 run immediately; re-run 01 anytime to
refresh it.

Neither yfinance nor interest-rate-models is a dependency of svi-py:
the calibration API takes a plain callable `T -> r(T)` and never asks
where the curve came from. Both packages are used only by script 01,
which persists the fitted zero curve densely in the snapshot metadata
-- the offline scripts reconstruct r(T) from the file with numpy alone.

## No lookahead

A surface is a picture of the market at one instant. The examples
enforce that mechanically: script 01 records a single snapshot
timestamp, fetches quotes, spot, and rates in one pass, filters
staleness using only information available at that timestamp, and
persists everything to disk. Scripts 02-05 read the files and compute
time-to-expiry against the recorded timestamp -- never the wall clock
-- so nothing downstream can accidentally use information from after
the snapshot. The same discipline applies to any production use:
calibrate only on data observable at the valuation time.

## Implied-vol inversion methods

The Black-Scholes price has no closed-form inverse; every IV is the
output of an inversion algorithm:

- **Black-76 root-finding** -- the textbook route (Newton/Brent on the
  pricing formula). Fine for illustration; fragile near intrinsic.
- **"Let's Be Rational" (Jackel, Wilmott 2015)** -- the de facto
  standard: rational approximations plus at most two Householder
  steps reach full machine precision at roughly the cost of two price
  evaluations. [Paper](http://www.jaeckel.org/LetsBeRational.pdf)
  ([journal](https://doi.org/10.1002/wilm.10395)),
  [reference implementation](https://github.com/vollib/lets_be_rational).
  **This is what svi-py uses internally**: all inversion goes through
  `py_vollib`, whose engine is `py_lets_be_rational`.
- **Volfi (Schadner)** -- newer still: an *explicit*, non-iterative
  inverse via a generalized-inverse-Gaussian quantile representation
  with an optional single Halley refinement. By the author of the
  DirectSVI parametrization shipped in this library.
  [Paper](https://arxiv.org/abs/2604.24480),
  [code](https://github.com/wol-fi/volfi). See also
  [fast-vollib](https://arxiv.org/abs/2604.27210) for vectorized LBR
  at scale.

## Maintenance policy

These examples are part of the API contract: whenever the public API
changes, the examples and the documentation must change in the same
pull request. CI-visible drift between `examples/` and the library is
a bug.

Data note: Yahoo Finance data is delayed and for personal/research use
under Yahoo's terms; the committed sample snapshot is included solely
to make the examples reproducible.
