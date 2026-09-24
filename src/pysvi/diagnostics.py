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

from . import _kernels
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


# ── Economic vs mathematical arbitrage (issue #31) ───────────────────

#: Classification labels for arbitrage findings.
CLASS_EXTRAPOLATION = "extrapolation_risk"
CLASS_QUOTE_CONSISTENT = "quote_consistent"
CLASS_EXECUTABLE = "executable"
CLASS_MATHEMATICAL = "mathematical"


@dataclass(frozen=True)
class ArbitrageClassification:
    """Economic classification of one slice's arbitrage findings.

    The numerical diagnostics report mathematical evidence; this layer
    says what that evidence MEANS: a violation outside the quoted
    strike range is extrapolation risk (the model's wings, not a
    trade); inside the range it is ``executable`` when a static
    butterfly built from the quoted crossed prices (buy at ask, sell
    at bid) has negative cost, ``quote_consistent`` when the violation
    disappears inside the bid/ask uncertainty, and ``mathematical``
    when no quote band is available to decide.
    """

    maturity: Optional[float]
    finding: str          # "butterfly" | "lee" | "none"
    k_violation: Optional[float]
    classification: Optional[str]
    detail: str

    def __str__(self) -> str:
        T = f"T={self.maturity:g}" if self.maturity is not None else "slice"
        if self.finding == "none":
            return f"{T}: clean"
        return f"{T}: {self.finding} violation -> {self.classification} ({self.detail})"


def _butterfly_cost_from_quotes(g, F, T, k_v):
    """Worst-case (buy-at-ask, sell-at-bid) cost of the static butterfly
    at the three quoted strikes nearest the violation; negative cost is
    an executable arbitrage. Forward-normalized undiscounted Black
    prices -- the sign is invariant to discounting and F scaling."""
    black = _kernels.resolve("black_call")
    q = g.dropna(subset=["iv_bid", "iv_ask"]).sort_values("strike")
    if len(q) < 3:
        return None
    K = q["strike"].to_numpy(dtype=float)
    k_q = np.log(K / F)
    j = int(np.clip(np.searchsorted(k_q, k_v), 1, len(K) - 2))
    K1, K2, K3 = K[j - 1], K[j], K[j + 1]
    def price(i, side):
        iv = float(q.iloc[i]["iv_" + side])
        return black(float(k_q[i]), iv * iv * T)
    # (K3-K2) C(K1) - (K3-K1) C(K2) + (K2-K1) C(K3) >= 0 arbitrage-free
    return (
        (K3 - K2) * price(j - 1, "ask")
        - (K3 - K1) * price(j, "bid")
        + (K2 - K1) * price(j + 1, "ask")
    )


def classify_arbitrage(surface, panel=None) -> tuple:
    """Classify a surface's arbitrage findings economically.

    Runs the diagnostics on the quoted range (as ``diagnose`` does) and
    classifies each slice's butterfly/Lee finding:

    * ``extrapolation_risk`` -- the violation sits outside the slice's
      quoted strike range: a property of the model's wings, not a
      constructible trade.
    * ``executable`` -- inside the quoted range AND a static butterfly
      built from the panel's bid/ask quotes around the violation has
      negative worst-case cost (buy wings at ask, sell body at bid).
    * ``quote_consistent`` -- inside the quoted range but the
      worst-case butterfly cost is non-negative: the violation lives
      within bid/ask uncertainty.
    * ``mathematical`` -- inside the quoted range, but no ``panel``
      with iv_bid/iv_ask was given to decide executability.

    Parameters
    ----------
    surface : VolSurface
        A fitted surface with a fit report (for the quoted ranges).
    panel : pd.DataFrame, optional
        Quote panel with iv_bid/iv_ask columns (as OptionChain
        produces) for the executability test.

    Returns
    -------
    tuple of ArbitrageClassification, one per fitted slice.
    """
    if surface.fit_report is None:
        raise ValueError("classify_arbitrage requires a surface fit report")
    report = surface.diagnose().arbitrage
    ranges = {
        s.maturity: (s.k_min, s.k_max)
        for s in surface.fit_report.slices if s.k_min is not None
    }
    out = []
    for sl in report.slices:
        rng = ranges.get(sl.maturity)
        if sl.ok:
            out.append(ArbitrageClassification(sl.maturity, "none", None, None, "no violation"))
            continue
        finding = "butterfly" if not sl.butterfly_free else "lee"
        k_v = sl.min_density_k if finding == "butterfly" else (
            sl.k_min if abs(sl.left_wing_slope) > abs(sl.right_wing_slope) else sl.k_max
        )
        if rng is None or k_v is None or k_v < rng[0] or k_v > rng[1]:
            out.append(ArbitrageClassification(
                sl.maturity, finding, k_v, CLASS_EXTRAPOLATION,
                "violation outside the quoted strike range",
            ))
            continue
        if finding == "lee":
            out.append(ArbitrageClassification(
                sl.maturity, finding, k_v, CLASS_MATHEMATICAL,
                "asymptotic wing bound; no static quoted portfolio tests it",
            ))
            continue
        if panel is None or "iv_bid" not in getattr(panel, "columns", ()):
            out.append(ArbitrageClassification(
                sl.maturity, finding, k_v, CLASS_MATHEMATICAL,
                "no bid/ask panel supplied to test executability",
            ))
            continue
        g = panel[np.isclose(panel["maturity"], sl.maturity)]
        F = float(g["implied_forward"].iloc[0]) if len(g) else None
        cost = (
            _butterfly_cost_from_quotes(g, F, float(sl.maturity), float(k_v))
            if F else None
        )
        if cost is None:
            out.append(ArbitrageClassification(
                sl.maturity, finding, k_v, CLASS_MATHEMATICAL,
                "insufficient two-sided quotes around the violation",
            ))
        elif cost < 0:
            out.append(ArbitrageClassification(
                sl.maturity, finding, k_v, CLASS_EXECUTABLE,
                f"static butterfly at quoted prices costs {cost:.3e} < 0",
            ))
        else:
            out.append(ArbitrageClassification(
                sl.maturity, finding, k_v, CLASS_QUOTE_CONSISTENT,
                f"worst-case butterfly cost {cost:.3e} >= 0: inside bid/ask uncertainty",
            ))
    return tuple(out)
