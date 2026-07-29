# Parametrizations

All parametrizations work in total variance space:

$$w(k) = \sigma^2(k) \cdot T$$

where $k = \log(K/F)$ is log-moneyness.

Every model implements the same two-method interface (`Parametrization` ABC):

- `calibrate(k, w_target, **kwargs)` — fit parameters from log-moneyness and observed total variance; returns a parameter `dict` or `None` on failure.
- `total_variance(k, params)` — evaluate the fitted surface.

Instances come from the factory:

```python
from pysvi import get_model, ArbitrageFreedom

model = get_model("svi")                                  # default QUASI constraints
model = get_model("sabr", ArbitrageFreedom.NO_BUTTERFLY)  # with density check
```

The factory accepts `"svi"`, `"ssvi"`, `"essvi"`, `"jumpwings"` (or `"jw"`), `"directsvi"` (or `"dsvi"`), and `"sabr"` (case-insensitive).

## Choosing a model

| Model | Free params / slice | Extra inputs | Best for |
|-------|--------------------:|--------------|----------|
| {doc}`svi` | 5 | — | Maximum flexibility, liquid equity smiles |
| {doc}`ssvi` | 2 | `theta` | Butterfly-arbitrage-free by construction |
| {doc}`essvi` | 4 (global) | `theta`, `theta_ref` | Realistic calendar skew across maturities |
| {doc}`jumpwings` | 5 | `T` | Trader-interpretable parameters (wings, ATM) |
| {doc}`directsvi` | 6 (closed-form) | — | Speed: no iterative optimisation |
| {doc}`sabr` | 3 (β fixed) | `T`, `F`, `beta` | Interest-rate and FX smiles; dynamic model |

Each model page follows the same structure: overview, model equations, parameter table, usage, arbitrage behaviour, and references.

```{toctree}
:maxdepth: 1

svi
ssvi
essvi
jumpwings
directsvi
sabr
```
