# svi-py

[![CI/CD](https://github.com/marwinsteiner/pysvi/actions/workflows/python-publish.yml/badge.svg)](https://github.com/marwinsteiner/pysvi/actions/workflows/python-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/svi-py)](https://pypi.org/project/svi-py/)
[![Python](https://img.shields.io/pypi/pyversions/svi-py)](https://pypi.org/project/svi-py/)
[![Downloads](https://static.pepy.tech/badge/svi-py)](https://pepy.tech/project/svi-py)
[![codecov](https://codecov.io/gh/marwinsteiner/pysvi/branch/main/graph/badge.svg)](https://codecov.io/gh/marwinsteiner/pysvi)
[![Docs](https://readthedocs.org/projects/pysvi/badge/?version=latest)](https://pysvi.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Stochastic volatility inspired (SVI) parametrizations of the implied volatility surface in Python — plus the SABR stochastic volatility model.

`svi-py` calibrates smooth, arbitrage-aware total variance surfaces from panels of European option prices: raw SVI, SSVI, eSSVI, jump-wings, DirectSVI, and SABR behind one interface, with configurable no-arbitrage constraints and a full data-preparation pipeline.

**Full documentation: [pysvi.readthedocs.io](https://pysvi.readthedocs.io)**

## Installation

```bash
pip install svi-py
```

Requires Python >= 3.13.

## Quick start

You need a DataFrame with columns for strike, implied vol, time to maturity, and implied forward:

```python
from pysvi import get_model, calibrate_slice, apply_slice

# df_slice: single-maturity cross-section with columns
#   strike, iv, maturity, implied_forward
model = get_model("svi")
params = calibrate_slice(df_slice, model)

fitted = apply_slice(df_slice, params, model)
print(fitted[["strike", "iv", "fitted_iv", "residual_iv"]])
```

The factory accepts `"svi"`, `"ssvi"`, `"essvi"`, `"jumpwings"` (or `"jw"`), `"directsvi"` (or `"dsvi"`), and `"sabr"`. Some models take extra per-slice arguments (`theta` for SSVI/eSSVI, `T` for jump-wings, `T`/`F`/`beta` for SABR) — see the [documentation](https://pysvi.readthedocs.io) for each parametrization's formulas, parameters, and usage, plus arbitrage-constraint options and the input-preparation helpers.

## Contributing

Contributions, bug reports, and feature requests are welcome. Open an issue or submit a PR on [GitHub](https://github.com/marwinsteiner/pysvi). See the [contributing guide](https://pysvi.readthedocs.io/en/latest/contributing.html).

**Wanted: the original Gamma-Vanna-Volga paper.** The Gamma-Vanna-Volga parametrization is something of a holy grail in the quant vol surface literature and would be a great addition to this library. If you have a copy of the original paper, please send it to [marwin.steiner@gmail.com](mailto:marwin.steiner@gmail.com).

## License

MIT
