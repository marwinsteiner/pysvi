# Calibration pipeline

All iterative models calibrate via L-BFGS-B (bounded quasi-Newton) with automatic Nelder-Mead fallback; DirectSVI solves a closed-form eigenvalue problem instead.

## Per-slice pipeline

The pipeline for a single maturity slice is:

1. **`prepare_slice`**: extracts $T$, $F$, computes $k = \log(K/F)$ and $w = \sigma_{\mathrm{mkt}}^2 T$, filters invalid data, clips extreme moneyness.
2. **`model.calibrate`**: minimises the mean squared error between model and market total variance, plus penalty terms:

$$\min_{\text{params}}\; \mathrm{MSE}\big(w_{\mathrm{model}}(k),\, w_{\mathrm{target}}\big) + \text{penalties}$$

3. **`apply_slice`**: evaluates the fitted surface and computes fitted vols and residuals:

$$\sigma_{\mathrm{fit}}(k) = \sqrt{\frac{w(k)}{T}}$$

## Input preparation helpers

If you're starting from raw option prices rather than implied vols:

### Implied volatilities

`compute_ivs_vectorized` computes Black-Scholes-Merton implied vols from option mid-prices via `py_vollib`. Failures (e.g. below-intrinsic prices) come back as `NaN`.

### Implied forwards

`calculate_implied_forward` estimates the forward price from put-call parity:

$$F = K + e^{rT}(C - P)$$

### OTM leg selection

`choose_leg` selects the OTM leg for cleaner vol quotes — calls for $K \geq F$, puts for $K < F$. OTM options have higher liquidity and no intrinsic-value noise, which improves calibration stability.

### Slice validation

`prepare_slice` rejects slices with fewer than `min_points` (default 5) valid strikes after cleaning, and clips log-moneyness to $[-10, 10]$ to prevent optimizer divergence from wing noise.
