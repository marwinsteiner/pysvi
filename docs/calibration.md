# Calibration pipeline

All iterative models calibrate via L-BFGS-B (bounded quasi-Newton) with automatic Nelder-Mead fallback; DirectSVI solves a closed-form eigenvalue problem instead.

## Calibration controls

Every iterative model accepts the same optional keyword arguments through `calibrate` / `calibrate_slice`:

```python
params = calibrate_slice(
    df_slice, model,
    objective="vega_weighted",       # residual space
    loss="soft_l1",                  # robust aggregation
    initialization="multi_start",    # start strategy
)
```

### Objectives (residual spaces)

- `"total_variance"` (default) — residuals in $w$; the historical behaviour.
- `"implied_vol"` — residuals in total volatility $\sqrt{w}$ (equivalent to implied-vol residuals up to a per-slice constant).
- `"price"` — residuals in forward-normalized undiscounted Black call prices.
- `"vega_weighted"` — volatility residuals weighted by Black vega computed from the market quotes (normalized to mean one); approximates price errors while staying in vol space, and is usually the best practical default for real chains.
- `"bid_ask"` — zero loss inside the quoted band, distance outside it. Pass the band as total-variance arrays: `w_bid = iv_bid**2 * T`, `w_ask = iv_ask**2 * T`.

### Robust losses

`loss="l2"` (default), `"huber"`, `"soft_l1"`, or `"cauchy"`, following the `scipy.optimize.least_squares` convention: contribution $= f^2 \rho(r/f)$ with scale $f$. When `f_scale` is not given, it defaults to $1.4826 \times \mathrm{MAD}$ of the residuals at a **pilot l2 fit**, floored at $10^{-6}$ of the data scale — so genuine outliers stand out against the fitted noise level, and on clean data the robust losses reduce to l2. A single corrupted far-wing quote that visibly distorts an l2 SVI fit is largely ignored under `soft_l1` or `cauchy`.

### Initialisation

- `initialization="default"` — the per-model heuristics (unchanged behaviour).
- `"jump_wings"` (SVI and NaturalSVI) — a data-driven start read off the quotes: wing slopes from the outer 20% of strikes on each side, skew from their asymmetry, vertex from the minimum-variance strike.
- `"multi_start"` — a deterministic grid of starts (the default start plus skew and width variations, 16 total); each runs L-BFGS-B to tight tolerance and the best converged result wins. Raw SVI's landscape has genuine bad basins that a single start can fall into — multi-start is the recommended setting whenever fit quality matters more than the last millisecond, and it is cheap under the numba backend.

When any control is active the optimizer runs with tight tolerances (`ftol=1e-15`); the plain default path keeps scipy's defaults for backward-compatible fits.

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
