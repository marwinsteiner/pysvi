# src/pysvi/report.py
"""Calibration fit reports and the surface diagnostics wrapper.

A surface is the output of a calibration process, and the process itself
carries the evidence needed to decide whether the output is trustworthy.
`SurfaceFitReport` records that evidence per slice (status, quote
accounting, residual statistics, quoted domain) together with the
settings and provenance of the fit; `SurfaceDiagnostics` combines it
with the arbitrage diagnostics into one formatted, field-accessible
result block (in the spirit of scikit-learn reports).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

from .diagnostics import ArbitrageReport

#: Slice statuses recorded by the fitters.
SLICE_OK = "ok"
SLICE_FAILED = "failed"
SLICE_INSUFFICIENT = "insufficient_data"


def _pysvi_version() -> str:
    try:
        from importlib.metadata import version

        return version("svi-py")
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class SliceFitReport:
    """Fit evidence for a single maturity slice.

    Attributes
    ----------
    maturity : float
        Slice maturity in years.
    status : str
        "ok", "failed" (calibration did not converge), or
        "insufficient_data" (slice rejected before calibration).
    n_quotes : int
        Rows entering the slice.
    n_used : int
        Quotes surviving cleaning (finite, positive) and used in the fit.
    iv_rmse : float or None
        RMSE of fitted vs market implied vol on the used quotes.
    max_abs_iv_residual : float or None
        Largest absolute implied-vol residual on the used quotes.
    k_min, k_max : float or None
        Quoted log-moneyness range (the slice's domain of validity).
    """

    maturity: float
    status: str
    n_quotes: int
    n_used: int
    iv_rmse: Optional[float] = None
    max_abs_iv_residual: Optional[float] = None
    k_min: Optional[float] = None
    k_max: Optional[float] = None


def build_slice_report(
    maturity: float,
    status: str,
    n_quotes: int,
    k=None,
    iv_mkt=None,
    iv_fit=None,
) -> SliceFitReport:
    """Assemble a SliceFitReport, computing residual statistics when fitted."""
    if k is None:
        return SliceFitReport(float(maturity), status, int(n_quotes), 0)
    k = np.asarray(k, dtype=np.float64)
    if status != SLICE_OK or iv_mkt is None or iv_fit is None:
        return SliceFitReport(
            float(maturity), status, int(n_quotes), int(k.size),
            k_min=float(k.min()), k_max=float(k.max()),
        )
    resid = np.asarray(iv_mkt, dtype=np.float64) - np.asarray(iv_fit, dtype=np.float64)
    return SliceFitReport(
        float(maturity), status, int(n_quotes), int(k.size),
        iv_rmse=float(np.sqrt(np.mean(resid**2))),
        max_abs_iv_residual=float(np.max(np.abs(resid))),
        k_min=float(k.min()), k_max=float(k.max()),
    )


@dataclass(frozen=True)
class SurfaceFitReport:
    """Calibration provenance and per-slice fit evidence for a surface.

    Attributes
    ----------
    model : str
        Parametrization class name.
    objective, loss, initialization : str
        Calibration controls used for every slice.
    calendar_enforced : bool
        Whether cross-slice calendar chaining was active.
    backend : str
        "numba" or "numpy" at fit time.
    slices : tuple of SliceFitReport
        Per-slice evidence, ordered by maturity (failed slices included).
    created_utc : str
        Fit timestamp (ISO 8601, UTC).
    pysvi_version : str
        Library version that produced the fit.
    """

    model: str
    objective: str
    loss: str
    initialization: str
    calendar_enforced: bool
    backend: str
    slices: Tuple[SliceFitReport, ...]
    created_utc: str
    pysvi_version: str
    #: Ingestion accounting (populated by OptionChain.fit): raw quote
    #: rows rejected before the panel, mid-quote inversions that failed
    #: to a NaN implied vol, and expiries dropped whole. Zero when the
    #: surface was fitted from a prepared panel directly.
    n_rejected_quotes: int = 0
    n_failed_inversions: int = 0
    n_skipped_expiries: int = 0

    @property
    def n_ok(self) -> int:
        return sum(1 for s in self.slices if s.status == SLICE_OK)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.slices if s.status != SLICE_OK)

    @property
    def ok(self) -> bool:
        """True iff every slice in the input panel calibrated."""
        return self.n_failed == 0

    @property
    def n_quotes(self) -> int:
        return sum(s.n_quotes for s in self.slices)

    @property
    def n_used(self) -> int:
        return sum(s.n_used for s in self.slices)

    def quoted_range(self) -> Optional[Tuple[float, float]]:
        """Union of the quoted log-moneyness ranges across fitted slices."""
        lows = [s.k_min for s in self.slices if s.k_min is not None]
        highs = [s.k_max for s in self.slices if s.k_max is not None]
        if not lows:
            return None
        return min(lows), max(highs)

    def summary(self) -> str:
        """Formatted fit-report block."""
        rows = []
        for s in self.slices:
            rmse = f"{s.iv_rmse:.2e}" if s.iv_rmse is not None else "-"
            mx = f"{s.max_abs_iv_residual:.2e}" if s.max_abs_iv_residual is not None else "-"
            rng = (
                f"[{s.k_min:+.3f}, {s.k_max:+.3f}]"
                if s.k_min is not None else "-"
            )
            flag = "" if s.status == SLICE_OK else "  <-- ATTENTION"
            rows.append(
                f"  {s.maturity:<8.4g} {s.status:<18} {s.n_quotes:>6} {s.n_used:>6}"
                f" {rmse:>10} {mx:>10}  {rng}{flag}"
            )
        cal = "enforced" if self.calendar_enforced else "not enforced"
        rejected = ""
        if self.n_rejected_quotes or self.n_failed_inversions or self.n_skipped_expiries:
            rejected = (
                f"Ingestion:       {self.n_rejected_quotes} quotes rejected / "
                f"{self.n_failed_inversions} inversions failed / "
                f"{self.n_skipped_expiries} expiries skipped\n"
            )
        header = (
            "SurfaceFitReport\n"
            "================\n"
            f"Model:           {self.model:<16} Backend:   {self.backend}\n"
            f"Objective:       {self.objective:<16} Loss:      {self.loss}\n"
            f"Initialization:  {self.initialization:<16} Calendar:  {cal}\n"
            f"Slices:          {self.n_ok} ok / {self.n_failed} failed or rejected\n"
            f"Quotes:          {self.n_quotes} in / {self.n_used} used\n"
            + rejected +
            "\n"
            f"  {'T':<8} {'status':<18} {'quotes':>6} {'used':>6}"
            f" {'iv RMSE':>10} {'max|res|':>10}  k-range\n"
        )
        footer = f"\nFitted {self.created_utc} | svi-py {self.pysvi_version}"
        return header + "\n".join(rows) + footer

    def __str__(self) -> str:
        return self.summary()


#: Failure-handling modes shared by the ingestion and fit entry points.
#: strict: the first bad input raises with its location. warn (default):
#: problems are logged and recorded. lenient: problems are filtered
#: silently but still recorded -- nothing ever disappears without a count.
FAILURE_MODES = ("strict", "warn", "lenient")


def validate_mode(mode: str) -> str:
    if mode not in FAILURE_MODES:
        raise ValueError(
            f"unknown mode {mode!r}; choose from {FAILURE_MODES}"
        )
    return mode


def build_surface_report(
    slices: Tuple[SliceFitReport, ...],
    model: str,
    model_kwargs: dict,
    calendar_enforced: bool,
) -> SurfaceFitReport:
    """Assemble a SurfaceFitReport with settings and provenance."""
    from . import _kernels

    return SurfaceFitReport(
        model=model,
        objective=str(model_kwargs.get("objective", "total_variance")),
        loss=str(model_kwargs.get("loss", "l2")),
        initialization=str(model_kwargs.get("initialization", "default")),
        calendar_enforced=calendar_enforced,
        backend="numba" if _kernels.numba_enabled() else "numpy",
        slices=tuple(slices),
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pysvi_version=_pysvi_version(),
    )


@dataclass(frozen=True)
class SurfaceDiagnostics:
    """Fit report and arbitrage diagnostics in one result block.

    Returned by :meth:`VolSurface.diagnose`. Fields are individually
    accessible; ``str()`` renders the combined formatted block.
    """

    fit: Optional[SurfaceFitReport]
    arbitrage: ArbitrageReport

    @property
    def ok(self) -> bool:
        """True iff every slice fitted and no arbitrage was detected."""
        fit_ok = self.fit.ok if self.fit is not None else True
        return fit_ok and self.arbitrage.ok

    def __str__(self) -> str:
        parts = []
        if self.fit is not None:
            parts.append(self.fit.summary())
        else:
            parts.append(
                "SurfaceFitReport\n================\n"
                "not available (surface constructed directly from parameters)"
            )
        parts.append(
            "\nArbitrage diagnostics\n=====================\n"
            + str(self.arbitrage)
        )
        verdict = "OK" if self.ok else "ATTENTION REQUIRED"
        parts.append(f"\nOverall: {verdict}")
        return "\n".join(parts)
