# SSVI (Surface SVI)

## Overview

Gatheral & Jacquier (2014). Reduces to 2 free parameters per slice by fixing the ATM total variance, and guarantees no butterfly arbitrage by construction for each fixed $\theta$.

## Model

$$w(k;\theta) = \frac{\theta}{2}\left[1 + \rho\,\varphi(\theta)\,k + \sqrt{\left(\varphi(\theta)\,k + \rho\right)^2 + 1 - \rho^2}\right]$$

where the curvature function is

$$\varphi(\theta) = \frac{\eta}{\sqrt{\theta}}$$

## Parameters

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| $\theta$ | ATM total variance (fixed input) | $\theta > 0$ |
| $\rho$ | skew | $\|\rho\| < 1$ |
| $\eta$ | curvature scale | $\eta > 0$ |

## Usage

```python
import numpy as np
from pysvi import get_model, calibrate_slice

model = get_model("ssvi")
theta = float(np.nanmin(df_slice["iv"] ** 2 * df_slice["maturity"]))
params = calibrate_slice(df_slice, model, theta=theta)
# params: {'rho', 'eta', 'theta', 'forward'}
```

## Arbitrage behaviour

No butterfly arbitrage by construction for fixed $\theta$. The `NO_BUTTERFLY` flag adds an explicit numerical check on top; `NO_CALENDAR` is available for cross-slice consistency — see {doc}`../arbitrage`.

## References

- Gatheral, J., Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces." *Quantitative Finance* 14(1).
