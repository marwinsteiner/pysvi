# src/pysvi/context.py
"""MarketContext: one coherent source of numeraire and time conventions.

A surface stores a rate and per-slice forwards, and maturities are bare
year-fraction floats. Nothing in those pieces says which day-count
produced T = 0.5, or guarantees that discounting, forwards and time all
came from the same convention. ``MarketContext`` is that single source
of truth: a valuation time, a rate view, a dividend view, a spot, and a
day-count convention, from which year fractions, discount factors and
fallback forwards are all derived coherently. Pass it to
``OptionChain.from_dataframe(context=...)`` (with real expiry DATES in
the panel) and every downstream number shares one convention by
construction; mixing a context with separately supplied rate/spot
inputs is loudly rejected.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .calibration import _rate_at

__all__ = ["MarketContext", "DAY_COUNTS"]

#: Supported day-count conventions.
DAY_COUNTS = ("ACT/365F", "ACT/360", "ACT/365.25", "BUS/252")


@dataclass(frozen=True)
class MarketContext:
    """Coherent numeraire, day-count and calendar conventions.

    Attributes
    ----------
    valuation_time : timestamp-like
        The single "now" every year fraction is measured from.
    spot : float, optional
        Underlying spot at valuation time (forward fallback only; real
        forwards come from put-call parity as always).
    rate : float, curve, model, or callable
        Continuously compounded zero rates in any form the library
        accepts (flat float, ``irm.DiscountCurve``, an interest-rate
        model, or T -> r(T)).
    dividend_yield : float, curve, model, or callable
        Continuous dividend view, same forms.
    day_count : str
        One of ``ACT/365F`` (default), ``ACT/360``, ``ACT/365.25``,
        ``BUS/252`` (business days over 252; supply ``holidays``).
    holidays : sequence of dates, optional
        Holiday calendar for BUS/252.
    """

    valuation_time: object
    spot: Optional[float] = None
    rate: object = 0.0
    dividend_yield: object = 0.0
    day_count: str = "ACT/365F"
    holidays: Optional[Sequence] = None

    def __post_init__(self):
        if self.day_count not in DAY_COUNTS:
            raise ValueError(
                f"unknown day_count {self.day_count!r}; choose from {DAY_COUNTS}"
            )
        object.__setattr__(
            self, "valuation_time", pd.Timestamp(self.valuation_time)
        )

    # ── Time ─────────────────────────────────────────────────────────

    def year_fraction(self, when):
        """Year fraction from valuation_time to ``when`` (scalar or
        array of timestamps/date strings), under this day count."""
        ts = pd.to_datetime(when)
        scalar = not isinstance(ts, (pd.DatetimeIndex, pd.Series))
        idx = pd.DatetimeIndex([ts]) if scalar else pd.DatetimeIndex(ts)
        if self.day_count == "BUS/252":
            hol = (
                np.array([np.datetime64(pd.Timestamp(h).date()) for h in self.holidays])
                if self.holidays is not None else None
            )
            start = np.datetime64(self.valuation_time.date())
            days = np.busday_count(
                start, idx.values.astype("datetime64[D]"),
                holidays=hol if hol is not None else [],
            )
            out = days / 252.0
        else:
            denom = {"ACT/365F": 365.0, "ACT/360": 360.0,
                     "ACT/365.25": 365.25}[self.day_count]
            delta = (idx - self.valuation_time).total_seconds() / 86400.0
            out = np.asarray(delta) / denom
        out = np.asarray(out, dtype=np.float64)
        return float(out[0]) if scalar else out

    # ── Numeraire ────────────────────────────────────────────────────

    def rate_at(self, T):
        """Continuously compounded zero rate r(T) under this context."""
        return _rate_at(self.rate, T)

    def dividend_at(self, T):
        return _rate_at(self.dividend_yield, T)

    def discount(self, T) -> float:
        """Discount factor D(T) = exp(-r(T) T). ``T`` may also be a
        date, resolved through :meth:`year_fraction` first."""
        if not np.isscalar(T) or isinstance(T, str):
            pass
        if isinstance(T, str) or hasattr(T, "toordinal") or isinstance(T, pd.Timestamp):
            T = self.year_fraction(T)
        T_arr = np.asarray(T, dtype=np.float64)
        r = np.asarray(self.rate_at(T_arr))
        out = np.exp(-r * T_arr)
        return float(out) if np.ndim(T) == 0 else out

    def forward(self, T) -> float:
        """Fallback forward spot * exp((r - q) T); real forwards come
        from put-call parity in the chain."""
        if self.spot is None:
            raise ValueError("MarketContext.forward requires spot")
        if isinstance(T, str) or hasattr(T, "toordinal") or isinstance(T, pd.Timestamp):
            T = self.year_fraction(T)
        r = float(np.asarray(self.rate_at(T)))
        q = float(np.asarray(self.dividend_at(T)))
        return float(self.spot) * float(np.exp((r - q) * T))
