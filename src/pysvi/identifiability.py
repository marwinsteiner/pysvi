# src/pysvi/identifiability.py
"""Parameter identifiability and uncertainty for calibrated slices.

A fitted smile can be excellent while its parameters are barely
determined: near-indistinguishable SVI smiles arise from very different
(a, b, rho, m, sigma), especially on narrow strike ranges where the
wings are unconstrained. Stable fitted implied vol is NOT stable fitted
parameters -- a signal built on parameter changes can be pure optimizer
noise. This module quantifies that:

* :func:`condition_number` -- conditioning of the (column-scaled)
  parameter Jacobian dw/dtheta at the quotes.
* :func:`parameter_uncertainty` -- Gauss-Newton covariance at the
  optimum, residual-scaled, as per-parameter standard errors and a
  correlation matrix.
* :func:`identifiability_report` -- the two combined into a formatted
  block flagging poorly identified parameters and near-degenerate
  parameter pairs.

The Jacobian comes from :meth:`Parametrization.param_jacobian`:
analytic for the SVI family (raw, natural, SSVI), central finite
differences elsewhere.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .models import Parametrization

__all__ = [
    "condition_number", "parameter_uncertainty", "identifiability_report",
    "ParameterUncertainty", "IdentifiabilityReport",
]


def condition_number(J: NDArray[np.float64]) -> float:
    """Condition number of the column-scaled parameter Jacobian.

    Columns are scaled to unit norm first, so the number measures the
    geometry of the parameter directions (how close to collinear they
    are at the quotes), not their units. Large values mean some
    parameter combination moves the fitted smile almost not at all --
    the optimizer could trade those parameters against each other
    freely. Returns inf for a rank-deficient Jacobian.
    """
    J = np.asarray(J, dtype=np.float64)
    norms = np.linalg.norm(J, axis=0)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        return float("inf")
    s = np.linalg.svd(J / norms, compute_uv=False)
    if s[-1] <= 0.0:
        return float("inf")
    return float(s[0] / s[-1])


@dataclass(frozen=True)
class ParameterUncertainty:
    """Gauss-Newton parameter uncertainty at a calibrated optimum.

    Attributes
    ----------
    names : tuple of str
        Free-parameter names, in Jacobian column order.
    values : tuple of float
        Fitted values.
    std_errors : tuple of float
        Per-parameter standard errors (sqrt of the covariance
        diagonal); inf where the fit is under-determined.
    correlation : ndarray
        p x p parameter correlation matrix.
    dof : int
        Residual degrees of freedom, n_points - n_params.
    rss : float
        Residual sum of squares in total-variance space.
    """

    names: Tuple[str, ...]
    values: Tuple[float, ...]
    std_errors: Tuple[float, ...]
    correlation: NDArray[np.float64]
    dof: int
    rss: float


def _uncertainty_with_workings(model, params, k, w_target):
    """parameter_uncertainty plus the Jacobian and residuals it used,
    so identifiability_report does not recompute them (the FD Jacobian
    costs 2p total_variance evaluations for every model without an
    analytic override)."""
    k = np.asarray(k, dtype=np.float64)
    w_target = np.asarray(w_target, dtype=np.float64)
    names = tuple(model.free_params)
    values = tuple(float(params[name]) for name in names)
    J = model.param_jacobian(k, params)
    n, p = J.shape
    dof = n - p
    resid = model.total_variance(k, params) - w_target
    rss = float(np.dot(resid, resid))
    if dof <= 0:
        u = ParameterUncertainty(
            names, values, tuple(float("inf") for _ in names),
            np.full((p, p), np.nan), dof, rss,
        )
        return u, J, resid
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(J.T @ J)
    diag = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(diag, diag)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0.0, cov / denom, np.nan)
    u = ParameterUncertainty(
        names, values, tuple(float(d) for d in diag), corr, dof, rss,
    )
    return u, J, resid


def parameter_uncertainty(
    model: Parametrization,
    params: Dict[str, float],
    k: NDArray[np.float64],
    w_target: NDArray[np.float64],
) -> ParameterUncertainty:
    """Standard errors and correlations from the Gauss-Newton
    approximation at the fitted optimum.

    The covariance is ``sigma^2 (J^T J)^+`` with ``sigma^2 = RSS /
    (n - p)`` in total-variance space and ``+`` the pseudo-inverse (so
    a rank-deficient Jacobian yields large-but-finite numbers in the
    identified directions and the report flags the degeneracy). With
    ``n <= p`` the fit is under-determined and every standard error is
    inf.
    """
    u, _, _ = _uncertainty_with_workings(model, params, k, w_target)
    return u


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Identifiability assessment of one calibrated slice.

    ``print(report)`` renders the formatted block; every field is
    individually accessible. ``ok`` is False when any parameter is
    flagged poorly identified or any pair is nearly degenerate.
    """

    model: str
    n_points: int
    k_min: float
    k_max: float
    rmse_w: float
    condition_number: float
    uncertainty: ParameterUncertainty
    poorly_identified: Tuple[str, ...]
    degenerate_pairs: Tuple[Tuple[str, str, float], ...]
    rel_threshold: float
    corr_threshold: float

    @property
    def ok(self) -> bool:
        return not self.poorly_identified and not self.degenerate_pairs

    def __str__(self) -> str:
        u = self.uncertainty
        lines = [
            "IdentifiabilityReport",
            "=====================",
            f"Model:            {self.model}",
            f"Quotes:           {self.n_points} on k in "
            f"[{self.k_min:+.3f}, {self.k_max:+.3f}]",
            f"Fit RMSE (w):     {self.rmse_w:.3e}",
            f"Condition number: {self.condition_number:.3g} (column-scaled Jacobian)",
            "",
            f"  {'param':<10} {'value':>12} {'std err':>12} {'rel err':>9}",
        ]
        for name, v, se in zip(u.names, u.values, u.std_errors):
            rel = se / abs(v) if v != 0 else float("inf")
            flag = "  <-- poorly identified" if name in self.poorly_identified else ""
            lines.append(
                f"  {name:<10} {v:>12.5g} {se:>12.3g} {rel:>8.1%}{flag}"
            )
        if self.degenerate_pairs:
            lines.append("")
            lines.append(
                f"Near-degenerate pairs (|corr| > {self.corr_threshold:g}):"
            )
            for a, b, c in self.degenerate_pairs:
                lines.append(f"  {a} ~ {b}: corr = {c:+.3f}")
        lines.append("")
        verdict = (
            "WELL IDENTIFIED" if self.ok else
            "ATTENTION: parameters are not individually trustworthy "
            "(the fitted smile may still be excellent)"
        )
        lines.append(f"Overall: {verdict}")
        return "\n".join(lines)


def identifiability_report(
    model: Parametrization,
    params: Dict[str, float],
    k: NDArray[np.float64],
    w_target: NDArray[np.float64],
    rel_threshold: float = 0.5,
    corr_threshold: float = 0.95,
) -> IdentifiabilityReport:
    """Assess how well the fitted parameters are determined by the data.

    Flags a parameter as poorly identified when its standard error
    exceeds ``rel_threshold`` of its magnitude, and a pair as
    near-degenerate when their correlation exceeds ``corr_threshold``
    in absolute value (the optimizer can trade one against the other
    with almost no change to the fitted smile). Typical trigger: a
    narrow strike range leaving the wing parameters unconstrained while
    the IV RMSE is tiny.

    Parameters
    ----------
    model : Parametrization
        The calibrated model instance.
    params : dict
        Its calibrated parameters (as returned by ``calibrate``).
    k, w_target : ndarray
        The quotes the fit used: log-moneyness and total variance
        (from ``prepare_slice``).
    rel_threshold : float, default 0.5
        Relative standard-error threshold for the per-parameter flag.
    corr_threshold : float, default 0.95
        Absolute-correlation threshold for the pair flag.
    """
    k = np.asarray(k, dtype=np.float64)
    w_target = np.asarray(w_target, dtype=np.float64)
    u, J, resid = _uncertainty_with_workings(model, params, k, w_target)
    poorly = tuple(
        name for name, v, se in zip(u.names, u.values, u.std_errors)
        if not np.isfinite(se) or se > rel_threshold * max(abs(v), 1e-12)
    )
    pairs = []
    p = len(u.names)
    for i in range(p):
        for j in range(i + 1, p):
            c = float(u.correlation[i, j])
            if np.isfinite(c) and abs(c) > corr_threshold:
                pairs.append((u.names[i], u.names[j], c))
    return IdentifiabilityReport(
        model=type(model).__name__,
        n_points=int(k.size),
        k_min=float(k.min()),
        k_max=float(k.max()),
        rmse_w=float(np.sqrt(np.mean(resid ** 2))),
        condition_number=condition_number(J),
        uncertainty=u,
        poorly_identified=poorly,
        degenerate_pairs=tuple(pairs),
        rel_threshold=rel_threshold,
        corr_threshold=corr_threshold,
    )
