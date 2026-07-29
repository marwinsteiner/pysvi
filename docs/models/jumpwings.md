# Jump-Wings

## Overview

The jump-wings parametrization (Gatheral 2004) re-expresses SVI in terms of financially interpretable quantities: ATM variance, ATM skew, wing slopes, and minimum variance. Same 5 degrees of freedom as raw SVI, but every parameter has a direct market meaning.

## Model

Jump-wings parameters map to raw SVI $(a, b, \rho, m, \sigma)$ via a bijection:

$$b = \frac{p_t + c_t}{2}, \quad \rho = 1 - \frac{p_t}{b}, \quad \beta = \rho - \frac{2\psi_t\sqrt{T}}{b}$$

$$\alpha = \mathrm{sgn}(\beta)\sqrt{\frac{1}{\beta^2} - 1}, \quad m = \frac{(v_t - \tilde{v}_t)\,T}{b\left[-\rho + \mathrm{sgn}(\alpha)\sqrt{1 + \alpha^2} - \alpha\sqrt{1 - \rho^2}\right]}$$

$$\sigma = |\alpha \cdot m|, \quad a = \tilde{v}_t \cdot T - b\,\sigma\sqrt{1 - \rho^2}$$

Evaluation then proceeds through the raw SVI formula.

## Parameters

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| $v_t$ | ATM variance $\sigma_{\mathrm{ATM}}^2$ | $v_t > 0$ |
| $\psi_t$ | ATM skew | bounded |
| $p_t$ | left (put) wing slope | $p_t \geq 0$ |
| $c_t$ | right (call) wing slope | $c_t \geq 0$ |
| $\tilde{v}_t$ | minimum implied variance | $\tilde{v}_t > 0$ |

## Usage

```python
from pysvi import get_model, calibrate_slice

model = get_model("jw")  # or "jumpwings"
T = float(df_slice["maturity"].iloc[0])
params = calibrate_slice(df_slice, model, T=T)
# params: {'v_t', 'psi_t', 'p_t', 'c_t', 'v_tilde_t', 'T', 'forward'}
```

## Arbitrage behaviour

Soft parameter bounds by default (`QUASI`); `NO_BUTTERFLY` and `NO_CALENDAR` penalties available via the raw-SVI conversion — see {doc}`../arbitrage`.

## References

- Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives."
