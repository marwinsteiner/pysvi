# SABR

## Overview

The SABR stochastic volatility model (Hagan, Kumar, Lesniewski & Woodward 2002) — the market standard for interest-rate (swaptions, caps/floors) and FX volatility smiles. Unlike the SVI family, SABR is a *dynamic model*: it specifies stochastic differential equations for the forward and its volatility, rather than a static smile shape.

## Model

Forward dynamics:

$$dF = \alpha F^\beta\, dW_1, \qquad d\alpha = \nu\, \alpha\, dW_2, \qquad d\langle W_1, W_2\rangle = \rho\, dt$$

Implied volatilities come from the Hagan et al. asymptotic expansion (the "HKLW formula"):

$$\sigma_B(K, F) = \frac{\alpha}{(FK)^{(1-\beta)/2}\, D(L)} \cdot \frac{z}{x(z)} \cdot \left(1 + \left[\frac{(1-\beta)^2\alpha^2}{24\,(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4\,(FK)^{(1-\beta)/2}} + \frac{2-3\rho^2}{24}\nu^2\right] T\right)$$

with log-moneyness ratio and wing-correction series

$$L = \log(F/K), \qquad D(L) = 1 + \frac{(1-\beta)^2}{24}L^2 + \frac{(1-\beta)^4}{1920}L^4$$

and the vol-of-vol transform

$$z = \frac{\nu}{\alpha}\,(FK)^{(1-\beta)/2}\, L, \qquad x(z) = \log\left(\frac{\sqrt{1-2\rho z+z^2}+z-\rho}{1-\rho}\right)$$

At the money the singular factor tends to one and is handled via Taylor expansion:

$$\lim_{K \to F}\, \frac{z}{x(z)} = 1$$

Total variance is then

$$w(k) = \sigma_B^2(k)\, T$$

## Parameters

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| $\alpha$ | initial volatility level | $\alpha > 0$ |
| $\beta$ | CEV exponent (**fixed**, not fitted) | $0 \leq \beta \leq 1$ |
| $\rho$ | spot/vol correlation | $\|\rho\| < 1$ |
| $\nu$ | vol-of-vol | $\nu \geq 0$ |

Following market practice, $\beta$ is fixed by convention rather than calibrated:

- $\beta = 1$ (lognormal) for FX and equity
- $\beta \approx 0.5$ for interest rates
- $\beta = 0$ (normal) for spread-like underlyings

## Usage

Calibration fits $(\alpha, \rho, \nu)$ per slice given $\beta$, $F$, $T$:

```python
from pysvi import get_model, calibrate_slice

model = get_model("sabr")
T = float(df_slice["maturity"].iloc[0])
F = float(df_slice["implied_forward"].iloc[0])
params = calibrate_slice(df_slice, model, T=T, F=F, beta=0.5)
# params: {'alpha', 'beta', 'rho', 'nu', 'F', 'T', 'forward'}
```

## Arbitrage behaviour

The HKLW expansion is accurate near the money and for moderate maturities, but is known to lose accuracy (and can even imply negative densities) for deep wings or very long maturities. `NO_BUTTERFLY` performs a numerical density check via finite differences — it is a guard, not a structural guarantee. `NO_CALENDAR` is supported via `w_prev` — see {doc}`../arbitrage`.

## References

- Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2002). "Managing Smile Risk", *Wilmott Magazine*.
- Obloj, J. (2008). "Fine-tune your smile: Correction to Hagan et al." — improved accuracy for $\beta < 1$.
