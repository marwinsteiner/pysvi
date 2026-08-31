# Arbitrage freeness

Arbitrage handling has two halves: *enforcement* during calibration (penalty flags, below) and *verification* of a fitted result (diagnostics). Constrained calibration is a soft penalty — always verify a fit you intend to rely on.

## Verifying a fit

`check_slice_arbitrage` inspects one calibrated slice; `check_arbitrage` inspects a set of slices across maturities:

```python
from pysvi import check_slice_arbitrage, check_arbitrage

report = check_slice_arbitrage(model, params, maturity=0.25)
report.ok               # single verdict
print(report)
# Slice (T=0.25):
#   Butterfly arbitrage: none (min g = 2.452e-01 at k = -2.0000)
#   Lee wing bounds:     satisfied (left slope = 0.1911, right slope = 0.0471, bound = 2)

report = check_arbitrage(model, [(0.25, params_1), (0.5, params_2)])
print(report)           # adds the calendar check between adjacent maturities
```

The reports carry numerical evidence, not just verdicts:

- **Butterfly**: minimum of the density factor $g(k)$ over the grid and the log-moneyness where it occurs (butterfly-free iff $g(k) \geq 0$).
- **Lee wing bounds**: total-variance wing slopes measured at the grid edges; the Lee moment formula requires $\limsup_{|k|\to\infty} w(k)/|k| \leq 2$ on each wing.
- **Calendar**: minimum of $w(k, T_{i+1}) - w(k, T_i)$ over adjacent maturity pairs and the $(k, T_i, T_{i+1})$ where it is attained (calendar-free iff non-negative).

The evaluation grid (`k_min`, `k_max`, `n_grid`) and tolerance are configurable; widen the grid for very wide smiles, since the wing slopes are measured at its edges.

Diagnostics build on the public derivative API: every parametrization exposes `dw_dk(k, params)`, `d2w_dk2(k, params)` and `density(k, params)` — analytic for the SVI family (raw SVI, SSVI, eSSVI, jump-wings), central finite differences for SABR and DirectSVI.

## Enforcement during calibration

Every parametrization accepts an `arbitrage_condition` argument controlling how strictly no-arbitrage is enforced during calibration. The options are flags that can be combined with `|`:

```python
from pysvi import get_model, ArbitrageFreedom

# Default: soft parameter bounds only
model = get_model("svi")  # ArbitrageFreedom.QUASI

# Enforce no butterfly arbitrage (non-negative density)
model = get_model("svi", ArbitrageFreedom.NO_BUTTERFLY)

# Enforce no calendar spread arbitrage (non-decreasing total variance in T)
model = get_model("ssvi", ArbitrageFreedom.NO_CALENDAR)

# Enforce both
model = get_model("svi", ArbitrageFreedom.NO_BUTTERFLY | ArbitrageFreedom.NO_CALENDAR)
```

## `QUASI` (default)

Soft parameter-bound constraints only: $b > 0$, $|\rho| < 1$, $\sigma > 0$. Enforced via bounded optimisation and penalty terms. Fast, and usually sufficient for liquid underlyings.

## `NO_BUTTERFLY`

Enforces non-negative call price density $g(k) \geq 0$ across strikes, where:

$$g(k) = \left(1 - \frac{k\,w'(k)}{2\,w(k)}\right)^2 - \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac{1}{4}\right) + \frac{w''(k)}{2}$$

Butterfly arbitrage exists whenever $g(k) < 0$ for some $k$. The calibrator evaluates $g$ on a fine grid and penalises violations.

Model-specific notes:

- **SSVI / eSSVI** already guarantee $g(k) \geq 0$ by their functional form; this flag adds an explicit numerical check.
- **SABR** uses finite-difference derivatives for $g$ (the Hagan expansion has no tractable closed-form $w''$); the check is a numerical guard, not a structural guarantee.
- **DirectSVI** does not support this flag (closed-form fit, nothing to penalise).

## `NO_CALENDAR`

Enforces non-decreasing total variance in maturity: $w(k, T_2) \geq w(k, T_1)$ for $T_2 > T_1$ at every $k$. This is a cross-slice condition. Pass the prior (shorter-maturity) slice's total variance via the `w_prev` keyword argument to `calibrate`:

```python
# After calibrating the first slice:
w_prev = model.total_variance(k_grid, params_first_slice)

# Calibrate the next slice with calendar constraint:
params_next = model.calibrate(k, w_target, w_prev=w_prev)
```
