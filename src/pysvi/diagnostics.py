# src/pysvi/diagnostics.py
"""First-class arbitrage diagnostics for calibrated parametrizations.

Verifies fitted slices and surfaces rather than trusting that
penalty-constrained calibration succeeded:

* butterfly arbitrage — non-negative risk-neutral density g(k)
* Lee wing bounds — total-variance wing slopes at most 2 [Lee 2004]
* calendar arbitrage — total variance non-decreasing in maturity

`check_slice_arbitrage` works on a single (model, params) pair;
`check_arbitrage` on a set of slices across maturities. Both return
report dataclasses that carry the numerical evidence (minima, locations,
grid) and render a human-readable summary via ``str()``.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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
    butterfly_free : bool
        True iff min g(k) >= -tol on the grid.
    min_density : float
        Minimum of the density factor g(k) over the grid.
    min_density_k : float
        Log-moneyness at which the minimum density occurs.
    lee_free : bool
        True iff both wing slopes are <= LEE_BOUND + tol.
    left_wing_slope : float
        dw/d|k| at the left grid edge (put wing).
    right_wing_slope : float
        dw/dk at the right grid edge (call wing).
    max_lee_violation : float
        max(wing slope - LEE_BOUND, 0) over both wings.
    k_min, k_max : float
        Evaluation grid bounds.
    n_grid : int
        Number of grid points.
    """

    maturity: Optional[float]
    butterfly_free: bool
    min_density: float
    min_density_k: float
    lee_free: bool
    left_wing_slope: float
    right_wing_slope: float
    max_lee_violation: float
    k_min: float
    k_max: float
    n_grid: int

    @property
    def ok(self) -> bool:
        """True iff no violation of any per-slice condition."""
        return self.butterfly_free and self.lee_free

    def __str__(self) -> str:
        label = f" (T={self.maturity:g})" if self.maturity is not None else ""
        bf = "none" if self.butterfly_free else "VIOLATION"
        lee = "satisfied" if self.lee_free else "VIOLATION"
        return (
            f"Slice{label}:\n"
            f"  Butterfly arbitrage: {bf}"
            f" (min g = {self.min_density:.3e} at k = {self.min_density_k:.4f})\n"
            f"  Lee wing bounds:     {lee}"
            f" (left slope = {self.left_wing_slope:.4f},"
            f" right slope = {self.right_wing_slope:.4f}, bound = {LEE_BOUND:g})"
        )


@dataclass(frozen=True)
class ArbitrageReport:
    """Arbitrage evidence for a set of calibrated slices across maturities.

    Attributes
    ----------
    slices : list of SliceArbitrageReport
        Per-slice reports, ordered by maturity.
    calendar_free : bool
        True iff total variance is non-decreasing in maturity at every
        grid point (within tol). Trivially True for a single slice.
    min_calendar_margin : float
        Minimum of w(k, T_next) - w(k, T) over adjacent maturity pairs
        and grid points; negative values are violations. +inf when fewer
        than two slices.
    min_calendar_k : float or None
        Log-moneyness of the worst calendar margin.
    min_calendar_pair : (float, float) or None
        The (T, T_next) pair attaining the worst margin.
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


def check_slice_arbitrage(
    model: Parametrization,
    params: Dict[str, float],
    maturity: Optional[float] = None,
    k_min: float = -2.0,
    k_max: float = 2.0,
    n_grid: int = 801,
    tol: float = 1e-8,
) -> SliceArbitrageReport:
    """Check a single calibrated slice for butterfly arbitrage and Lee bounds.

    Butterfly: evaluates the density factor g(k) on a uniform grid and
    records its minimum and location. Lee bounds: measures the total
    variance wing slopes dw/d|k| at the grid edges as a proxy for the
    asymptotic slopes (for SVI-family models the asymptotes are reached
    quickly; widen the grid for very wide smiles).

    Parameters
    ----------
    model : Parametrization
        Model instance matching the params dict.
    params : dict
        Calibrated parameters (as returned by ``model.calibrate``).
    maturity : float, optional
        Recorded in the report for labelling; not used in the checks.
    k_min, k_max : float, default -2.0, 2.0
        Evaluation grid bounds in log-moneyness.
    n_grid : int, default 801
        Grid resolution.
    tol : float, default 1e-8
        Numerical tolerance for violations.

    Returns
    -------
    SliceArbitrageReport
    """
    k = np.linspace(k_min, k_max, n_grid)
    g = model.density(k, params)
    i = int(np.nanargmin(g))
    min_g = float(g[i])

    right_slope = float(model.dw_dk(np.array([k_max]), params)[0])
    left_slope = float(-model.dw_dk(np.array([k_min]), params)[0])
    max_violation = max(left_slope - LEE_BOUND, right_slope - LEE_BOUND, 0.0)

    return SliceArbitrageReport(
        maturity=maturity,
        butterfly_free=min_g >= -tol,
        min_density=min_g,
        min_density_k=float(k[i]),
        lee_free=max_violation <= tol,
        left_wing_slope=left_slope,
        right_wing_slope=right_slope,
        max_lee_violation=max_violation,
        k_min=k_min,
        k_max=k_max,
        n_grid=n_grid,
    )


def check_arbitrage(
    model: Parametrization,
    slices: Iterable[Tuple[float, Dict[str, float]]],
    k_min: float = -2.0,
    k_max: float = 2.0,
    n_grid: int = 801,
    tol: float = 1e-8,
) -> ArbitrageReport:
    """Check calibrated slices across maturities for arbitrage.

    Runs :func:`check_slice_arbitrage` on every slice and additionally
    verifies the calendar condition w(k, T2) >= w(k, T1) for T2 > T1 on
    a shared grid, for every adjacent maturity pair.

    Parameters
    ----------
    model : Parametrization
        Model instance matching all params dicts.
    slices : iterable of (maturity, params)
        Calibrated slices; a dict {maturity: params} also works via
        ``.items()`` upstream. Sorted internally by maturity.
    k_min, k_max, n_grid, tol
        As in :func:`check_slice_arbitrage`.

    Returns
    -------
    ArbitrageReport

    Examples
    --------
    >>> report = check_arbitrage(model, [(0.25, p1), (0.5, p2)])
    >>> report.ok
    True
    >>> print(report)  # doctest: +SKIP
    """
    ordered = sorted(slices, key=lambda item: item[0])
    if not ordered:
        raise ValueError("check_arbitrage requires at least one (maturity, params) slice")

    reports = [
        check_slice_arbitrage(
            model, params, maturity=T,
            k_min=k_min, k_max=k_max, n_grid=n_grid, tol=tol,
        )
        for T, params in ordered
    ]

    calendar_free = True
    min_margin = float("inf")
    min_k: Optional[float] = None
    min_pair: Optional[Tuple[float, float]] = None
    if len(ordered) >= 2:
        k = np.linspace(k_min, k_max, n_grid)
        w_prev = model.total_variance(k, ordered[0][1])
        for (t_prev, _), (t_next, params_next) in zip(ordered, ordered[1:]):
            w_next = model.total_variance(k, params_next)
            margin = w_next - w_prev
            j = int(np.nanargmin(margin))
            if float(margin[j]) < min_margin:
                min_margin = float(margin[j])
                min_k = float(k[j])
                min_pair = (t_prev, t_next)
            w_prev = w_next
        calendar_free = min_margin >= -tol

    return ArbitrageReport(
        slices=reports,
        calendar_free=calendar_free,
        min_calendar_margin=min_margin,
        min_calendar_k=min_k,
        min_calendar_pair=min_pair,
    )
