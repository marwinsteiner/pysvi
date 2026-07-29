# eSSVI (Extended SSVI)

## Overview

Extends SSVI with maturity-dependent skew via a $\rho(\theta)$ term structure. Adds 4 parameters globally and enables realistic calendar skew evolution across maturities.

## Model

The total variance formula is the same as SSVI but with $\rho \to \rho(\theta)$:

$$w(k;\theta) = \frac{\theta}{2}\left[1 + \rho(\theta)\,\varphi(\theta)\,k + \sqrt{\left(\varphi(\theta)\,k + \rho(\theta)\right)^2 + 1 - \rho(\theta)^2}\right]$$

with the skew term structure

$$\rho(\theta) = \mathrm{clip}\left(\rho_0 + \rho_1 \left(\frac{\theta}{\theta_{\mathrm{ref}}}\right)^\alpha,\; -1,\; 1\right)$$

and curvature $\varphi(\theta) = \eta / \sqrt{\theta}$. Here $\theta_{\mathrm{ref}}$ is a reference ATM total variance (typically the median across slices) that normalises the power law.

## Parameters

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| $\rho_0$ | base skew level | $\|\rho_0\| < 1$ |
| $\rho_1$ | skew term-structure slope | bounded |
| $\alpha$ | power-law exponent | bounded |
| $\eta$ | curvature scale | $\eta > 0$ |
| $\theta$ | slice ATM total variance (fixed input) | $\theta > 0$ |
| $\theta_{\mathrm{ref}}$ | reference ATM total variance (fixed input) | $\theta_{\mathrm{ref}} > 0$ |

## Usage

```python
import numpy as np
from pysvi import get_model, calibrate_slice

model = get_model("essvi")
theta = float(np.nanmin(df_slice["iv"] ** 2 * df_slice["maturity"]))
params = calibrate_slice(df_slice, model, theta=theta, theta_ref=theta)
# params: {'rho0', 'rho1', 'alpha', 'eta', 'theta', 'theta_ref', 'rho_theta', 'forward'}
```

## Arbitrage behaviour

Inherits SSVI's per-slice butterfly guarantee for each fixed $\theta$; `NO_BUTTERFLY` / `NO_CALENDAR` numerical checks available — see {doc}`../arbitrage`.

## References

- Hendriks, S., Martini, C. (2019). "The extended SSVI volatility surface." *Journal of Computational Finance*.
