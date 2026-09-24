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

With `objective="bid_ask"`, panels carrying `iv_bid`/`iv_ask` columns (as `OptionChain` produces) get their per-slice `w_bid`/`w_ask` bands derived automatically — rows with a missing or crossed band degenerate to fit-to-mid. Slices that fail to calibrate are skipped with a warning. A surface can also be assembled directly from calibrated slices: `VolSurface(model, {T: params, ...})`, where each params dict carries `'forward'` (as `calibrate_slice` returns).

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
- `"monotone_cubic"` (any model, at least two slices) — a shape-preserving cubic (PCHIP, Fritsch–Carlson) in $T$ at fixed log-moneyness across **all** fitted slices. Exact at fitted maturities, monotone in $T$ wherever the fitted slices are (calendar-free slices stay calendar-free between expiries), and continuously differentiable in maturity — see Differentiability below.

Forwards interpolate log-linearly in $T$ (piecewise-constant forward rate). All evaluation and pricing methods (`iv`, `total_variance`, `price`, Greeks, `atm_vol`, `skew`, `curvature`) accept any maturity in range; `params(T)` remains exact-slice-only, and `slice_at(T)` between slices requires the `"theta"` method.

```python
surface = calibrate_surface(df, model="ssvi")
surface.iv(100.0, 1.37)        # interpolated maturity
surface.price(95.0, 1.37, "put")
```

### Differentiability and Dupire-readiness

`surface.regularity` declares the smoothness guarantee in maturity: `"C0"` for the linear blend and the theta method (continuous, but $\partial w/\partial T$ jumps at every fitted slice), `"C1"` for `"monotone_cubic"`. A C0 surface is **pricing-ready** — implied vols, prices, and sticky-strike Greeks are all well defined — but **not Dupire-ready**: local volatility, forward variance, and PDE coefficients consume $\partial w/\partial T$, which would be discontinuous exactly at the traded expiries.

With `interp_method="monotone_cubic"` the maturity derivative exists and is continuous everywhere in the fitted range, exposed directly:

```python
surface = VolSurface.fit(df, model="svi", interp_method="monotone_cubic")
surface.regularity        # "C1"
surface.dw_dT(k, T)       # the Dupire numerator, any T in range
```

`dw_dT` on a C0 surface raises rather than return a one-sided number that would be silently wrong at the knots. Smoothness in strike comes from the model itself and is analytic for the SVI family under every method.

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

## Variance events

Real term structures contain known jumps — earnings, FOMC, CPI, elections. Generic maturity interpolation smooths across them, silently asserting all term-structure curvature is continuous variance. `VarianceEvent(time, variance, label)` makes the jump explicit:

```python
from pysvi import VarianceEvent, implied_event_variances

surface = VolSurface.fit(df, events=[VarianceEvent(0.4, 0.006, "earnings")])
```

Total variance decomposes as $w(k,T) = w_{cont}(k,T) + \sum_{t_e \le T} v_e$: fitting subtracts each expiry's cumulative event variance (slices store the **continuous** component; events inconsistent with the quoted variance are rejected), interpolation acts on the continuous component, and evaluation adds the events back on the correct side of each event time — so the surface reproduces the jump exactly instead of smearing it, and the calendar diagnostics raise no false violation across an event. Events serialize with the surface; `dw_dT` reports the continuous derivative (the jump is a jump). `implied_event_variances(df, events)` reports the quoted ATM-variance jump straddling each event as an upper bound on its variance.

## Evaluation status

Every slice records its quoted strike range on the fit report, and evaluation can say whether a number is market information or model wing:

```python
iv, status = surface.iv(K, T, return_status=True)
# status per point: "observed" | "interpolated" | "extrapolated"
```

`"observed"` means a fitted maturity inside that slice's quoted range; `"interpolated"` sits between fitted maturities inside the bracketing slices' joint quoted range; everything else is `"extrapolated"` — the model's wings, not the market. See also the economic arbitrage classification on the {doc}`arbitrage <arbitrage>` page.

## Fit reports and diagnostics

A surface is the output of a calibration process, and the process carries the evidence needed to trust the output. `fit` records that evidence on the surface:

```python
surface.fit_report        # per-slice status, quote accounting, residuals, settings, provenance
print(surface.diagnose()) # fit report + arbitrage diagnostics in one formatted block
```

`fit_report` lists every slice of the input panel — including slices that failed to calibrate or were rejected for insufficient data — with quote counts (in vs used), implied-vol RMSE and max residual, and the quoted log-moneyness range. The report also records the calibration settings (objective, loss, initialization, backend) and provenance (svi-py version, fit timestamp). `report.ok` is False whenever any slice of the panel did not make it into the surface, so partial fits cannot pass silently.

`diagnose()` combines the fit report with `check_arbitrage`, run by default on the quoted strike range (the surface's domain of validity) rather than the wide default grid, and renders a single scikit-learn-style result block; every field remains individually accessible on the returned object. Surfaces constructed directly from parameter dicts have `fit_report=None` and `diagnose()` reports arbitrage only.

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
