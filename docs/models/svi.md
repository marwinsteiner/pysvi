# Raw SVI

## Overview

The original Gatheral (2004) parametrization with 5 free parameters per slice. Maximum flexibility; the workhorse for liquid equity smiles.

## Model

$$w(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right]$$

## Parameters

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| $a$ | overall variance level | $a \geq 0$ |
| $b$ | slope / curvature scale | $b > 0$ |
| $\rho$ | skew (correlation) | $\|\rho\| < 1$ |
| $m$ | log-moneyness shift | unconstrained |
| $\sigma$ | vol-of-vol (smile width) | $\sigma > 0$ |

## Usage

```python
from pysvi import get_model, calibrate_slice

model = get_model("svi")
params = calibrate_slice(df_slice, model)
# params: {'a', 'b', 'rho', 'm', 'sigma', 'forward'}
```

No extra keyword arguments are required.

## Arbitrage behaviour

No automatic arbitrage guarantees beyond soft parameter bounds (`QUASI`). Combine with `NO_BUTTERFLY` and/or `NO_CALENDAR` for penalty-based enforcement — see {doc}`../arbitrage`.

## References

- Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives."
