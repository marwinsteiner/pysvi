# Natural SVI

## Overview

The natural SVI parametrization (Gatheral & Jacquier 2014). Same 5 degrees of freedom as raw SVI, connected by an explicit bijection, but the parameters map more directly to ATM level, skew, and curvature — often better behaved in calibration, and useful as an initialisation coordinate system for raw SVI.

## Model

$$w(k) = \Delta + \frac{\omega}{2}\left[1 + \zeta\rho\,(k - \mu) + \sqrt{\left(\zeta(k - \mu) + \rho\right)^2 + 1 - \rho^2}\right]$$

The bijection to raw SVI $(a, b, \rho, m, \sigma)$:

$$a = \Delta + \frac{\omega(1-\rho^2)}{2}, \qquad b = \frac{\omega\zeta}{2}, \qquad m = \mu - \frac{\rho}{\zeta}, \qquad \sigma = \frac{\sqrt{1-\rho^2}}{\zeta}$$

and its inverse:

$$\zeta = \frac{\sqrt{1-\rho^2}}{\sigma}, \qquad \omega = \frac{2b\sigma}{\sqrt{1-\rho^2}}, \qquad \mu = m + \frac{\rho\sigma}{\sqrt{1-\rho^2}}, \qquad \Delta = a - b\sigma\sqrt{1-\rho^2}$$

Both directions are exposed as `natural_to_raw` and `raw_to_natural`.

## Parameters

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| $\Delta$ | vertical variance shift | unconstrained |
| $\mu$ | log-moneyness translation | unconstrained |
| $\rho$ | skew (correlation) | $\|\rho\| < 1$ |
| $\omega$ | overall variance scale | $\omega > 0$ |
| $\zeta$ | curvature / smile-width scale | $\zeta > 0$ |

## Usage

```python
from pysvi import get_model, calibrate_slice

model = get_model("natural")  # or "nsvi"
params = calibrate_slice(df_slice, model)
# params: {'delta', 'mu', 'rho', 'omega', 'zeta', 'forward'}
```

No extra keyword arguments are required. Converting a fit between conventions:

```python
from pysvi import natural_to_raw, raw_to_natural

raw = natural_to_raw(params["delta"], params["mu"], params["rho"],
                     params["omega"], params["zeta"])
```

## Arbitrage behaviour

Identical to raw SVI (the curves are the same family): soft parameter bounds by default (`QUASI`); `NO_BUTTERFLY` and `NO_CALENDAR` penalties supported via the raw-SVI conversion — see {doc}`../arbitrage`. Wing slopes for the Lee-bound diagnostics are the raw-SVI asymptotes $b(1 \mp \rho)$.

## References

- Gatheral, J., Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces." *Quantitative Finance* 14(1), section 3.
