# The VolSurface object

`VolSurface` is the fitted-surface abstraction: model → calibration → surface. It owns calibrated slices across maturities and exposes evaluation, diagnostics, and pricing — the object quant work actually consumes, rather than raw parameter dicts.

## Fitting

```python
from pysvi import VolSurface

surface = VolSurface.fit(df, model="svi", r=0.02)
```

`df` is a multi-expiry panel in the `calibrate_slice` schema (`strike`, `iv`, `maturity`, `implied_forward`). Every parametrization works through the same call — per-slice extras are derived automatically ($\theta$ per slice for SSVI/eSSVI with $\theta_{\mathrm{ref}}$ defaulting to the median, $T$ for jump-wings, $T$/$F$ for SABR with $\beta$ overridable via kwargs), and the calibration controls (`objective`, `loss`, `initialization`, …) pass through to every slice:

```python
surface = VolSurface.fit(df, model="ssvi", loss="soft_l1", initialization="multi_start")
```

Slices that fail to calibrate are skipped with a warning. A surface can also be assembled directly from calibrated slices: `VolSurface(model, {T: params, ...})`, where each params dict carries `'forward'` (as `calibrate_slice` returns).

## Calendar-aware fitting

`VolSurface.fit` calibrates slices independently. `calibrate_surface` owns the cross-maturity logic:

```python
from pysvi import calibrate_surface

surface = calibrate_surface(df, model="ssvi", enforce_calendar=True)
```

- expiries are ordered and calibrated oldest-first, with each fitted slice evaluated on the next slice's penalty grid and passed as `w_prev` automatically — the manual chaining the per-slice API requires;
- `enforce_calendar` adds `NO_CALENDAR` to the arbitrage condition, and for SSVI/eSSVI clips the per-slice ATM total variances $\theta(T)$ to be non-decreasing before fitting;
- for **eSSVI** the global term structure $(\rho_0, \rho_1, \alpha, \eta)$ is fitted *jointly across all slices* against the shared $\theta_{\mathrm{ref}}$ — every returned slice carries identical shape parameters, which is the point of the model;
- after fitting, SSVI-form slices are checked against the Gatheral-Jacquier sufficient no-butterfly bounds ($\theta\varphi(1+|\rho|) \leq 4$ and $\theta\varphi^2(1+|\rho|) \leq 4$) and one aggregated warning is logged for slices outside the proven-safe region (the bounds are conservative — verify with `check_arbitrage`).

Calibration controls pass through to every slice. DirectSVI is rejected when `enforce_calendar` is set (its closed-form fit has no penalty to enforce).

## Interpolation in maturity

Between fitted maturities the surface interpolates; beyond the fitted range it raises (no extrapolation). Two methods, chosen at construction via `interp_method`:

- `"total_variance"` (default, any model) — linear interpolation of $w(k)$ in $T$ at fixed log-moneyness. Model-agnostic, exact at fitted maturities, and calendar-free between two slices whenever they are ordered ($w$ of the blend lies between them at every $k$).
- `"theta"` (SSVI/eSSVI) — interpolates the ATM total variance $\theta(T)$ and shape parameters, yielding a genuine parametric slice at any maturity; `surface.slice_at(T)` returns its params dict. Under a joint eSSVI fit the shape parameters are shared, so only $\theta$ actually interpolates.

Forwards interpolate log-linearly in $T$ (piecewise-constant forward rate). All evaluation and pricing methods (`iv`, `total_variance`, `price`, Greeks, `atm_vol`, `skew`, `curvature`) accept any maturity in range; `params(T)` remains exact-slice-only, and `slice_at(T)` between slices requires the `"theta"` method.

```python
surface = calibrate_surface(df, model="ssvi")
surface.iv(100.0, 1.37)        # interpolated maturity
surface.price(95.0, 1.37, "put")
```

## Evaluation

```python
surface.maturities            # fitted maturities, ascending
surface.iv(strike, T)         # implied vol at absolute strikes
surface.total_variance(k, T)  # w(k) in log-moneyness
surface.forward(T)
surface.atm_vol(T)
surface.skew(T)               # dw/dk at k = 0
surface.curvature(T)          # d2w/dk2 at k = 0
surface.params(T)             # per-slice parameter dict
```

All strike/moneyness inputs are vectorized; scalar in, scalar out. Any maturity inside the fitted range works (see Interpolation below); maturities outside it raise.

## Verifying

```python
report = surface.check_arbitrage(k_data=k_observed)
report.ok
print(report)
```

forwards to the {doc}`arbitrage diagnostics <arbitrage>` across all slices (butterfly, Lee wing bounds, calendar). Prefer `k_data` so verification covers the quoted strike range: an unconstrained fit extrapolated far outside its data can legitimately fail the butterfly check out in the wings.

## Pricing and Greeks

A Black-76 layer on the slice forward, with a flat continuously compounded rate `r` set at construction:

$$C = e^{-rT}\left[F\,N(d_1) - K\,N(d_2)\right], \qquad d_1 = \frac{\log(F/K) + w/2}{\sqrt{w}}, \qquad d_2 = d_1 - \sqrt{w}$$

```python
surface.price(K, T, cp="call")
surface.delta(K, T, cp="call")   # forward delta, e^{-rT} N(d1)
surface.gamma(K, T)
surface.vega(K, T)               # per unit volatility
surface.theta(K, T, cp="call")   # per year of calendar time
```

Conventions: Greeks hold the implied volatility fixed (sticky-strike); delta and gamma are with respect to the forward; vega is per 1.00 of volatility; theta is annualized. Prices are verified against `py_vollib` and the Greeks against finite differences of the Black price in the test suite.

Deliberately out of scope here: American exercise, exotics, rate sensitivity, curve-aware discounting (curves are a later release; `r` is flat), and extrapolation beyond the fitted maturity range.
