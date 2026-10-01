# Parameter identifiability

A fitted smile can be excellent while its parameters are barely determined: near-indistinguishable SVI smiles arise from very different $(a, b, \rho, m, \sigma)$, especially on narrow strike ranges where the wings are unconstrained. **Stable fitted implied vol is not stable fitted parameters** — a signal built on day-over-day parameter changes can be pure optimizer noise. `pysvi.identifiability` quantifies this at any calibrated optimum.

## The report

```python
from pysvi import SVI, identifiability_report
from pysvi.calibration import prepare_slice

k, w, F = prepare_slice(df_slice)
model = SVI()
params = model.calibrate(k, w, initialization="multi_start")
print(identifiability_report(model, params, k, w))
```

```text
IdentifiabilityReport
=====================
Model:            SVI
Quotes:           15 on k in [-0.050, +0.050]
Fit RMSE (w):     2.397e-04
Condition number: 1.1e+05 (column-scaled Jacobian)

  param             value      std err   rel err
  a               0.01144         3.02 26390.6%  <-- poorly identified
  b               0.12117         7.55  6231.4%  <-- poorly identified
  ...

Near-degenerate pairs (|corr| > 0.95):
  a ~ b: corr = -1.000
  ...

Overall: ATTENTION: parameters are not individually trustworthy
(the fitted smile may still be excellent)
```

A parameter is flagged when its standard error exceeds `rel_threshold` (default 0.5) of its magnitude; a pair when their correlation exceeds `corr_threshold` (default 0.95) in absolute value — the optimizer can trade one against the other with almost no change to the fitted smile. `report.ok` summarizes; every field is individually accessible.

## The machinery

- `model.param_jacobian(k, params)` — the $n \times p$ sensitivity matrix $\partial w(k_i)/\partial \theta_j$ over the model's `free_params` (per-slice givens such as $\theta$, $T$, $F$, or a fixed $\beta$ are not free). Analytic for raw SVI, natural SVI, and SSVI; central finite differences elsewhere.
- `condition_number(J)` — of the column-scaled Jacobian: how close the parameter directions are to collinear at the quotes.
- `parameter_uncertainty(model, params, k, w)` — the Gauss-Newton covariance $\hat\sigma^2 (J^\top J)^+$ at the optimum, with $\hat\sigma^2 = \mathrm{RSS}/(n - p)$ in total-variance space, reported as standard errors and a correlation matrix. With $n \le p$ the fit is under-determined and every standard error is infinite.

## Quote-to-surface sensitivities

The same Gauss-Newton system answers the trader's question directly: *if this quote moves one vol point, what does the surface do?*

```python
from pysvi import quote_sensitivity, surface_sensitivity, iv_surface_sensitivity

S_theta = quote_sensitivity(model, params, k, w)            # dtheta/dquote, p x n
S_w     = surface_sensitivity(model, params, k, w, k_eval)  # dw(k_eval)/dquote
S_iv    = iv_surface_sensitivity(model, params, k, w, k_eval, T)  # vol-in, vol-out
```

These are implicit-function Jacobians at the optimum — `(J^T J)^+ J^T` propagated through `dw/dtheta` — so one linear solve replaces a recalibration per bump: hedging, P&L explain and scenario responses in vectorized form. Valid to first order; verified against bump-and-recalibrate in the test suite.

## What to do with it

- Widening the quoted strike range shrinks the uncertainties — often dramatically; the standard errors tell you whether today's chain supports the parameter you care about.
- If a signal needs stable parameters, prefer the well-identified combinations (e.g. ATM level and skew, which are read off the smile directly) over raw parameters, or switch to a lower-dimensional model (SSVI's two shape parameters identify far more sharply than raw SVI's five).
- The condition number is a fast health check inside pipelines; the full report is for the post-mortem.
