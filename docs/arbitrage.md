# Arbitrage freeness

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
