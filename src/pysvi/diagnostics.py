# src/pysvi/diagnostics.py
"""First-class arbitrage diagnostics for calibrated parametrizations.

Verifies fitted slices and surfaces rather than trusting that
penalty-constrained calibration succeeded:

* domain validity — w(k) must be finite and strictly positive before any
  other criterion is meaningful (g(k) is only defined for w > 0)
* butterfly arbitrage — non-negative risk-neutral density g(k)
* Lee wing bounds — total-variance wing slopes at most 2 [Lee 2004],
  using the model's closed-form asymptotic slopes where available
  (the SVI family) and a grid-edge measurement otherwise
* calendar arbitrage — total variance non-decreasing in maturity

`check_slice_arbitrage` works on a single (model, params) pair;
`check_arbitrage` on a set of slices across maturities. Both return
report dataclasses that carry the numerical evidence (minima, locations,
invalid-point counts, grid) and render a human-readable summary via
``str()``. A report never claims freedom from arbitrage it could not
evaluate: invalid or non-finite grid regions fail the check.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from .models import Parametrization

#: Lee moment bound on total-variance wing slopes: limsup w(k)/abs(k) <= 2.
LEE_BOUND = 2.0


@dataclass(frozen=True)
class SliceArbitrageReport:
    """Numerical arbitrage evidence for a single calibrated slice.

    Attributes
    ----------
    maturity : float or None
        Slice maturity, if provided.
    n_invalid : int
        Grid points where w is non-finite, w <= 0, or g is non-finite.
        Any invalid point fails the butterfly check: the criterion is
        undefined there, so freedom from arbitrage cannot be certified.
    min_total_variance : float
        Minimum of w(k) over the grid (NaN if w is nowhere finite).
    butterfly_free : bool
        True iff the whole grid is valid and min g(k) >= -tol.
    min_density : float
        Minimum of g(k) over the valid grid points (NaN if none).
    min_density_k : float
        Log-moneyness at which the minimum density occurs (NaN if none).
    lee_free : bool
        True iff both wing slopes are finite and <= LEE_BOUND + tol.
    left_wing_slope : float
        Left (put) wing slope dw/d(abs(k)).
    right_wing_slope : float
        Right (call) wing slope dw/dk.
    wing_slope_method : str
        "asymptotic" (closed form, SVI family) or "grid_edge" (measured
        at the grid boundary; underestimates the asymptote for convex w,
        so widen the grid for wide smiles).
    max_lee_violation : float
        max(wing slope - LEE_BOUND, 0) over both wings (NaN if a slope
        is non-finite).
    k_min, k_max : float
        Evaluation grid bounds.
    n_grid : int
        Number of grid points.
    """

    maturity: Optional[float]
    n_invalid: int
    min_total_variance: float
    butterfly_free: bool
    min_density: float
    min_density_k: float
    lee_free: bool
    left_wing_slope: float
    right_wing_slope: float
    wing_slope_method: str
    max_lee_violation: float
    k_min: float
    k_max: float
    n_grid: int

    @property
    def ok(self) -> bool:
        """True iff every per-slice condition was evaluable and passed."""
        return self.n_invalid == 0 and self.butterfly_free and self.lee_free

    def __str__(self) -> str:
        label = f" (T={self.maturity:g})" if self.maturity is not None else ""
        if not math.isfinite(self.min_density):
            bf = "NOT EVALUABLE"
        elif self.butterfly_free:
            bf = "none"
        else:
            bf = "VIOLATION"
        if not (math.isfinite(self.left_wing_slope) and math.isfinite(self.right_wing_slope)):
            lee = "NOT EVALUABLE"
        elif self.lee_free:
            lee = "satisfied"
        else:
            lee = "VIOLATION"
        lines = [f"Slice{label}:"]
        if self.n_invalid:
            lines.append(
                f"  Invalid grid points: {self.n_invalid}/{self.n_grid}"
                f" (w <= 0 or non-finite; min w = {self.min_total_variance:.3e})"
            )
        lines.append(
            f"  Butterfly arbitrage: {bf}"
            f" (min g = {self.min_density:.3e} at k = {self.min_density_k:.4f})"
        )
        lines.append(
            f"  Lee wing bounds:     {lee}"
            f" (left slope = {self.left_wing_slope:.4f},"
            f" right slope = {self.right_wing_slope:.4f},"
            f" bound = {LEE_BOUND:g}, {self.wing_slope_method})"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class ArbitrageReport:
    """Arbitrage evidence for a set of calibrated slices across maturities.

    Attributes
    ----------
    slices : list of SliceArbitrageReport
        Per-slice reports, ordered by maturity.
    calendar_free : bool
        True iff every adjacent maturity pair was evaluable and total
        variance is non-decreasing in maturity at every grid point
        (within tol). Trivially True for a single slice.
    min_calendar_margin : float
        Minimum of w(k, T_next) - w(k, T) over adjacent maturity pairs
        and grid points; negative values are violations. +inf when fewer
        than two slices; NaN when no pair was evaluable.
    min_calendar_k : float or None
        Log-moneyness of the worst calendar margin (None if none
        evaluable).
    min_calendar_pair : (float, float) or None
        The (T, T_next) pair attaining the worst margin (None if none
        evaluable).
    """

    slices: List[SliceArbitrageReport]
    calendar_free: bool
    min_calendar_margin: float
    min_calendar_k: Optional[float]
    min_calendar_pair: Optional[Tuple[float, float]]

    @property
    def ok(self) -> bool:
        """True iff every slice passes and there is no calendar arbitrage."""
        return self.calendar_free and all(s.ok for s in self.slices)

    def __str__(self) -> str:
        lines = [str(s) for s in self.slices]
        if len(self.slices) < 2:
            lines.append("Calendar arbitrage:    n/a (single slice)")
        elif self.min_calendar_pair is None:
            lines.append(
                "Calendar arbitrage:    NOT EVALUABLE (no finite margins)"
            )
        else:
            cal = "none" if self.calendar_free else "VIOLATION"
            t1, t2 = self.min_calendar_pair
            lines.append(
                f"Calendar arbitrage:    {cal}"
                f" (min dw = {self.min_calendar_margin:.3e}"
                f" at k = {self.min_calendar_k:.4f}, between T={t1:g} and T={t2:g})"
            )
        verdict = "ARBITRAGE-FREE" if self.ok else "ARBITRAGE DETECTED"
        return "\n".join(lines + [f"Overall:               {verdict}"])


def _resolve_grid(
    k_min: float, k_max: float, k_data
) -> Tuple[float, float]:
    """Grid bounds; k_data mirrors the calibration penalty grid policy."""
    if k_data is not None:
        k_data = np.asarray(k_data, dtype=np.float64)
        return float(k_data.min()) - 0.5, float(k_data.max()) + 0.5
    return k_min, k_max


def check_slice_arbitrage(
    model: Parametrization,
    params: Dict[str, float],
    maturity: Optional[float] = None,
    k_min: float = -2.0,
    k_max: float = 2.0,
    n_grid: int = 801,
    tol: float = 1e-8,
    k_data=None,
) -> SliceArbitrageReport:
    """Check a single calibrated slice for butterfly arbitrage and Lee bounds.

    Domain validity comes first: grid points where w(k) is non-finite,
    non-positive, or where g(k) is non-finite are counted as invalid, and
    any invalid point fails the butterfly check (freedom from arbitrage
    is never certified on an unevaluated region).

    Lee bounds use the model's closed-form asymptotic wing slopes where
    available (:meth:`Parametrization.wing_slopes`; the SVI family), and
    otherwise measure dw/dk at the grid edges — a proxy that
    underestimates the asymptote for convex w, so widen the grid for
    very wide smiles when the method reads "grid_edge".

    Parameters
    ----------
    model : Parametrization
        Model instance matching the params dict.
    params : dict
        Calibrated parameters (as returned by ``model.calibrate``).
    maturity : float, optional
        Recorded in the report for labelling; not used in the checks.
    k_min, k_max : float, default -2.0, 2.0
        Evaluation grid bounds in log-moneyness. Prefer ``k_data`` so
        verification covers the same domain calibration penalized.
    n_grid : int, default 801
        Grid resolution.
    tol : float, default 1e-8
        Numerical tolerance for violations. For models with
        finite-difference derivatives (SABR, DirectSVI) the density
        carries FD noise well above this; use a looser tol (e.g. 1e-4)
        there for marginal fits.
    k_data : array, optional
        Observed log-moneyness values. When given, the grid is
        [min(k_data) - 0.5, max(k_data) + 0.5] — the same policy the
        NO_BUTTERFLY/NO_CALENDAR calibration penalties use, so
        verification and enforcement cover the same domain.

    Returns
    -------
    SliceArbitrageReport
    """
    k_min, k_max = _resolve_grid(k_min, k_max, k_data)
    k = np.linspace(k_min, k_max, n_grid)
    # Degenerate parameters are this function's job to judge, not to warn
    # about: overflow/invalid warnings are suppressed, non-finite results
    # are counted as invalid points below.
    with np.errstate(all="ignore"):
        w = model.total_variance(k, params)
        g = model.density(k, params)

    valid = np.isfinite(w) & (w > 0) & np.isfinite(g)
    n_invalid = int(np.sum(~valid))
    min_w = float(np.nanmin(w)) if np.any(np.isfinite(w)) else float("nan")

    if np.any(valid):
        g_masked = np.where(valid, g, np.inf)
        i = int(np.argmin(g_masked))
        min_g = float(g[i])
        min_g_k = float(k[i])
    else:
        min_g = float("nan")
        min_g_k = float("nan")
    butterfly_free = n_invalid == 0 and min_g >= -tol

    slopes = model.wing_slopes(params)
    if slopes is not None:
        left_slope, right_slope = float(slopes[0]), float(slopes[1])
        method = "asymptotic"
    else:
        right_slope = float(model.dw_dk(np.array([k_max]), params)[0])
        left_slope = float(-model.dw_dk(np.array([k_min]), params)[0])
        method = "grid_edge"
    if math.isfinite(left_slope) and math.isfinite(right_slope):
        max_violation = max(left_slope - LEE_BOUND, right_slope - LEE_BOUND, 0.0)
        lee_free = max_violation <= tol
    else:
        max_violation = float("nan")
        lee_free = False

    return SliceArbitrageReport(
        maturity=maturity,
        n_invalid=n_invalid,
        min_total_variance=min_w,
        butterfly_free=butterfly_free,
        min_density=min_g,
        min_density_k=min_g_k,
        lee_free=lee_free,
        left_wing_slope=left_slope,
        right_wing_slope=right_slope,
        wing_slope_method=method,
        max_lee_violation=max_violation,
        k_min=k_min,
        k_max=k_max,
        n_grid=n_grid,
    )


def check_arbitrage(
    model: Parametrization,
    slices: Union[
        Mapping[float, Dict[str, float]],
        Iterable[Tuple[float, Dict[str, float]]],
    ],
    k_min: float = -2.0,
    k_max: float = 2.0,
    n_grid: int = 801,
    tol: float = 1e-8,
    k_data=None,
) -> ArbitrageReport:
    """Check calibrated slices across maturities for arbitrage.

    Runs :func:`check_slice_arbitrage` on every slice and additionally
    verifies the calendar condition w(k, T2) >= w(k, T1) for T2 > T1 on
    a shared grid, for every adjacent maturity pair. A pair with no
    finite margins (degenerate slices) fails the calendar check rather
    than passing silently.

    Parameters
    ----------
    model : Parametrization
        Model instance matching all params dicts.
    slices : mapping or iterable of (maturity, params)
        Calibrated slices, as a dict {maturity: params} or an iterable
        of pairs. Sorted internally by maturity; duplicate maturities
        are rejected (two fits of one expiry are not a calendar pair).
    k_min, k_max, n_grid, tol, k_data
        As in :func:`check_slice_arbitrage`.

    Returns
    -------
    ArbitrageReport

    Examples
    --------
    >>> report = check_arbitrage(model, {0.25: p1, 0.5: p2})
    >>> report.ok
    True
    >>> print(report)  # doctest: +SKIP
    """
    if isinstance(slices, Mapping):
        slices = slices.items()
    ordered = sorted(slices, key=lambda item: item[0])
    if not ordered:
        raise ValueError("check_arbitrage requires at least one (maturity, params) slice")
    maturities = [t for t, _ in ordered]
    if len(set(maturities)) != len(maturities):
        raise ValueError(
            "check_arbitrage requires distinct maturities; got duplicates "
            f"in {maturities} (two fits of one expiry are not a calendar pair)"
        )

    reports = [
        check_slice_arbitrage(
            model, params, maturity=T,
            k_min=k_min, k_max=k_max, n_grid=n_grid, tol=tol, k_data=k_data,
        )
        for T, params in ordered
    ]
    grid_min, grid_max = _resolve_grid(k_min, k_max, k_data)

    calendar_free = True
    min_margin = float("inf")
    min_k: Optional[float] = None
    min_pair: Optional[Tuple[float, float]] = None
    if len(ordered) >= 2:
        k = np.linspace(grid_min, grid_max, n_grid)
        with np.errstate(all="ignore"):
            w_by_slice = [
                model.total_variance(k, params) for _, params in ordered
            ]
        for idx, ((t_prev, _), (t_next, _)) in enumerate(zip(ordered, ordered[1:])):
            with np.errstate(invalid="ignore"):
                margin = w_by_slice[idx + 1] - w_by_slice[idx]
            finite = np.isfinite(margin)
            if not np.any(finite):
                calendar_free = False  # pair not evaluable: cannot certify
            else:
                masked = np.where(finite, margin, np.inf)
                j = int(np.argmin(masked))
                if float(margin[j]) < min_margin:
                    min_margin = float(margin[j])
                    min_k = float(k[j])
                    min_pair = (t_prev, t_next)
        if min_pair is None:
            min_margin = float("nan")
        elif min_margin < -tol:
            calendar_free = False

    return ArbitrageReport(
        slices=reports,
        calendar_free=calendar_free,
        min_calendar_margin=min_margin,
        min_calendar_k=min_k,
        min_calendar_pair=min_pair,
    )
