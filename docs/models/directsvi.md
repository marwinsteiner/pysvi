# DirectSVI

## Overview

A closed-form SVI calibration method (Schadner, forthcoming) that linearises the SVI equation by rewriting it as a conic section (hyperbola) in $(k, w)$ space. No iterative optimisation is needed, making this the fastest calibration method in the library.

## Model

The SVI curve is expressed as a conic:

$$z_0 k^2 + z_1 w^2 + z_2 kw + z_3 k + z_4 w + z_5 = 0$$

The 6 conic coefficients $z$ are found by solving a quadratically constrained eigenvalue problem (hyperbola constraint $z_2^2 - 4z_0 z_1 > 0$):

1. Build design matrices $D_2 = [k^2,\; w^2]$ and $D_1 = [kw,\; k,\; w,\; 1]$
2. Compute scatter matrices $S_{22}, S_{21}, S_{11}$
3. Solve $M\mathbf{a}_2 = \lambda\, C_1\mathbf{a}_2$ where $M = S_{22} - S_{21}S_{11}^{-1}S_{21}^\top$ and $C_1 = \begin{pmatrix}0 & -2\\-2 & 0\end{pmatrix}$
4. Select the eigenvector for the smallest positive eigenvalue; recover remaining coefficients via $\mathbf{a}_1 = -S_{11}^{-1}S_{21}^\top\mathbf{a}_2$

Evaluation solves the conic for $w$ given $k$ via the quadratic formula:

$$w = \frac{-(z_2 k + z_4) + \sqrt{(z_2 k + z_4)^2 - 4z_1(z_0 k^2 + z_3 k + z_5)}}{2 z_1}$$

## Parameters

| Parameter | Meaning |
|-----------|---------|
| $z_0$ – $z_5$ | Conic section coefficients (normalised so $z_1 = 1$) |

## Usage

```python
from pysvi import get_model, calibrate_slice

model = get_model("dsvi")  # or "directsvi"
params = calibrate_slice(df_slice, model)
# params: {'z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'forward'}
```

## Arbitrage behaviour

DirectSVI does not support penalty-based arbitrage enforcement (`NO_BUTTERFLY` / `NO_CALENDAR`) — the fit is closed-form, so there is no objective to penalise. Only `ArbitrageFreedom.QUASI` is meaningful; other flags are ignored with a warning.

## References

- Schadner, W. "Direct Fit for SVI Implied Volatilities", *Journal of Derivatives* (forthcoming). See also [`wol-fi/directSVI`](https://github.com/wol-fi/directSVI).
