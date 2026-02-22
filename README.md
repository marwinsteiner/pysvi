# svi-py

Stochastic volatility inspired (SVI) parametrizations of the implied volatility surface in Python.

Given a panel of contemporaneous European call and put option prices across strikes and maturities, `svi-py` calibrates smooth, arbitrage-aware total variance surfaces using the SVI family of models. It handles the full pipeline: implied vol extraction, forward estimation via put-call parity, OTM leg selection, and per-slice calibration with configurable no-arbitrage constraints.

## Installation

```bash
pip install svi-py
```

Requires Python >= 3.13.

## Quick start

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

For SSVI/eSSVI, pass the ATM total variance as an extra argument:

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

### Where do the inputs come from?

`svi-py` expects you to already have implied volatilities and forward prices. If you're starting from raw option prices, the library provides helpers:

- `compute_ivs_vectorized` computes Black-Scholes-Merton implied vols from option mid-prices via `py_vollib`.
- `calculate_implied_forward` estimates the forward price from put-call parity: $F = K + e^{rT}(C - P)$.
- `choose_leg` selects the OTM leg (calls for $K \geq F$, puts for $K < F$) for cleaner vol quotes.

You need a panel of **contemporaneous call and put option prices** across multiple strikes for at least one maturity. The richer the strike grid, the better the calibration.

## Parametrizations

All parametrizations work in total variance space: $w(k) = \sigma^2(k) \cdot T$, where $k = \log(K/F)$ is log-moneyness.

### Raw SVI

The original Gatheral (2004) parametrization with 5 free parameters:

$$w(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right]$$

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| $a$ | overall variance level | $a \geq 0$ |
| $b$ | slope / curvature scale | $b > 0$ |
| $\rho$ | skew (correlation) | $\|\rho\| < 1$ |
| $m$ | log-moneyness shift | unconstrained |
| $\sigma$ | vol-of-vol (smile width) | $\sigma > 0$ |

Maximum flexibility (5 degrees of freedom per slice). No automatic arbitrage guarantees beyond soft parameter bounds.

### SSVI (Surface SVI)

Gatheral & Jacquier (2014). Reduces to 2 free parameters per slice by fixing the ATM total variance $\theta = \sigma_{\text{ATM}}^2 T$:

$$w(k;\theta) = \frac{\theta}{2}\left[1 + \rho\,\varphi(\theta)\,k + \sqrt{\left(\varphi(\theta)\,k + \rho\right)^2 + 1 - \rho^2}\right]$$

where $\varphi(\theta) = \eta / \sqrt{\theta}$ is the curvature function.

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| $\theta$ | ATM total variance (fixed input) | $\theta > 0$ |
| $\rho$ | skew | $\|\rho\| < 1$ |
| $\eta$ | curvature scale | $\eta > 0$ |

Guarantees no butterfly arbitrage by construction for each fixed $\theta$.

### eSSVI (Extended SSVI)

Extends SSVI with maturity-dependent skew via a $\rho(\theta)$ term structure:

$$\rho(\theta) = \text{clip}\!\left(\rho_0 + \rho_1 \left(\frac{\theta}{\theta_{\text{ref}}}\right)^\alpha,\; -1,\; 1\right)$$

The total variance formula is the same as SSVI but with $\rho \to \rho(\theta)$. This adds 4 parameters globally ($\rho_0, \rho_1, \alpha, \eta$) and enables realistic calendar skew evolution across maturities. $\theta_{\text{ref}}$ is a reference ATM total variance (typically the median across slices) that normalises the power law.

### Jump-Wings

The jump-wings parametrization (Gatheral 2004) re-expresses SVI in terms of financially interpretable quantities:

| Parameter | Meaning |
|-----------|---------|
| $v_t$ | ATM variance $\sigma_{\text{ATM}}^2$ |
| $\psi_t$ | ATM skew |
| $p_t$ | left (put) wing slope, $p_t \geq 0$ |
| $c_t$ | right (call) wing slope, $c_t \geq 0$ |
| $\tilde{v}_t$ | minimum implied variance, $\tilde{v}_t > 0$ |

These map to raw SVI $(a, b, \rho, m, \sigma)$ via a bijection:

$$b = \frac{p_t + c_t}{2}, \quad \rho = 1 - \frac{p_t}{b}, \quad \beta = \rho - \frac{2\psi_t\sqrt{T}}{b}$$

$$\alpha = \text{sgn}(\beta)\sqrt{\frac{1}{\beta^2} - 1}, \quad m = \frac{(v_t - \tilde{v}_t)\,T}{b\left[-\rho + \text{sgn}(\alpha)\sqrt{1 + \alpha^2} - \alpha\sqrt{1 - \rho^2}\right]}$$

$$\sigma = |\alpha \cdot m|, \quad a = \tilde{v}_t \cdot T - b\,\sigma\sqrt{1 - \rho^2}$$

Same 5 degrees of freedom as raw SVI but with parameters that have direct market interpretation (wing slopes, ATM level, minimum variance).

## Arbitrage freeness

Every parametrization accepts an `arbitrage_condition` argument controlling how strictly no-arbitrage is enforced during calibration. The options are flags that can be combined with `|`:

```python
from pysvi import ArbitrageFreedom

# Default: soft parameter bounds only
model = get_model("svi")  # ArbitrageFreedom.QUASI

# Enforce no butterfly arbitrage (non-negative density)
model = get_model("svi", ArbitrageFreedom.NO_BUTTERFLY)

# Enforce no calendar spread arbitrage (non-decreasing total variance in T)
model = get_model("ssvi", ArbitrageFreedom.NO_CALENDAR)

# Enforce both
model = get_model("svi", ArbitrageFreedom.NO_BUTTERFLY | ArbitrageFreedom.NO_CALENDAR)
```

### `QUASI` (default)

Soft parameter-bound constraints only: $b > 0$, $|\rho| < 1$, $\sigma > 0$. Enforced via bounded optimisation and penalty terms. Fast, and usually sufficient for liquid underlyings.

### `NO_BUTTERFLY`

Enforces non-negative call price density $g(k) \geq 0$ across strikes, where:

$$g(k) = \left(1 - \frac{k\,w'(k)}{2\,w(k)}\right)^2 - \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac{1}{4}\right) + \frac{w''(k)}{2}$$

Butterfly arbitrage exists whenever $g(k) < 0$ for some $k$. The calibrator evaluates $g$ on a fine grid and penalises violations. Note that SSVI and eSSVI already guarantee $g(k) \geq 0$ by their functional form; this flag adds an explicit numerical check.

### `NO_CALENDAR`

Enforces non-decreasing total variance in maturity: $w(k, T_2) \geq w(k, T_1)$ for $T_2 > T_1$ at every $k$. This is a cross-slice condition. Pass the prior (shorter-maturity) slice's total variance via the `w_prev` keyword argument to `calibrate`:

```python
# After calibrating the first slice:
w_prev = model.total_variance(k_grid, params_first_slice)

# Calibrate the next slice with calendar constraint:
params_next = model.calibrate(k, w_target, w_prev=w_prev)
```

## Calibration details

All models calibrate via L-BFGS-B (bounded quasi-Newton) with automatic Nelder-Mead fallback. The pipeline for a single maturity slice is:

1. **`prepare_slice`**: extracts $T$, $F$, computes $k = \log(K/F)$ and $w = \sigma_{\text{mkt}}^2 T$, filters invalid data, clips extreme moneyness.
2. **`model.calibrate`**: minimises MSE$(w_{\text{model}}, w_{\text{target}})$ plus penalty terms.
3. **`apply_slice`**: evaluates the fitted surface, computes $\sigma_{\text{fit}} = \sqrt{w/T}$ and residuals.

The factory function `get_model(name)` accepts `"svi"`, `"ssvi"`, `"essvi"`, `"jumpwings"` (or `"jw"`).

## Contributing

Contributions, bug reports, and feature requests are welcome. Open an issue or submit a PR on [GitHub](https://github.com/marwinsteiner/pysvi).

**Wanted: the original Gamma-Vanna-Volga paper.** The Gamma-Vanna-Volga parametrization is something of a holy grail in the quant vol surface literature and would be a great addition to this library. If you have a copy of the original paper, please send it to [marwin.steiner@gmail.com](mailto:marwin.steiner@gmail.com).

## License

MIT
