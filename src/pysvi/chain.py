# src/pysvi/chain.py
"""Option-chain ingestion: raw call/put quotes to a calibration-ready panel.

`OptionChain` formalises the preprocessing every user otherwise
reassembles by hand: mid computation, implied forwards from put-call
parity, OTM leg selection, Black-76 implied-vol inversion, and bid/ask
IV bands. The result is the `calibrate_slice` panel schema plus
`iv_bid`/`iv_ask` columns, and `chain.fit(...)` goes straight to a
:class:`~pysvi.surface.VolSurface`.

IV inversion is Black-76 on the implied forward (dividends and repo are
embedded in the forward), so neither spot nor a dividend assumption is
needed when both legs are quoted; a spot plus rate/dividend inputs serve
only as the forward fallback for expiries missing put-call pairs.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from py_lets_be_rational.exceptions import (AboveMaximumException,
                                             BelowIntrinsicException)
from py_vollib.black.implied_volatility import implied_volatility as _black_iv

from .calibration import _rate_at, calculate_implied_forward, choose_leg
from .models import ArbitrageFreedom, Parametrization
from .report import validate_mode
from .surface import VolSurface, calibrate_surface

RateLike = Union[float, "callable"]


def _invert_iv(price, F, K, r, T, flag) -> float:
    """Black-76 implied vol from a discounted option price; NaN on failure."""
    if not (np.isfinite(price) and price > 0):
        return float("nan")
    try:
        return float(_black_iv(float(price), float(F), float(K), float(r), float(T), flag))
    except (BelowIntrinsicException, AboveMaximumException,
            ValueError, ZeroDivisionError, OverflowError):
        return float("nan")


class OptionChain:
    """A cleaned option chain ready for surface calibration.

    Build with :meth:`from_dataframe`; access the calibration panel via
    :attr:`panel` (the `calibrate_slice` schema: ``strike``, ``iv``,
    ``maturity``, ``implied_forward``, plus ``iv_bid``/``iv_ask``), and
    fit a surface with :meth:`fit`.
    """

    def __init__(self, panel: pd.DataFrame, rate: RateLike = 0.0,
                 rejections: Optional[dict] = None) -> None:
        self._panel = panel.reset_index(drop=True)
        self.rate = rate
        #: Ingestion accounting: {"rejected_quotes", "failed_inversions",
        #: "skipped_expiries" (list of T)}. Populated by from_dataframe.
        self.rejections = rejections or {
            "rejected_quotes": 0, "failed_inversions": 0, "skipped_expiries": [],
        }

    @property
    def panel(self) -> pd.DataFrame:
        """Calibration panel (a copy)."""
        return self._panel.copy()

    @property
    def maturities(self) -> np.ndarray:
        return np.sort(self._panel["maturity"].unique())

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        strike: str = "strike",
        expiry: str = "expiry",
        cp: str = "cp",
        bid: str = "bid",
        ask: str = "ask",
        spot: Optional[float] = None,
        rate: RateLike = 0.0,
        dividend_yield: RateLike = 0.0,
        mode: str = "warn",
    ) -> "OptionChain":
        """Ingest raw call/put quotes into a calibration-ready chain.

        Per expiry: mids from bid/ask (rows with crossed, missing, or
        non-positive quotes are dropped), the implied forward as the
        median put-call-parity forward over strikes quoting both legs
        (falling back to ``spot * exp((r - q) T)`` when no pair exists
        and ``spot`` is given), OTM leg selection, and Black-76 implied
        vols for mid, bid, and ask on the selected leg. Quotes whose
        mid cannot be inverted become NaN ivs and are counted out by the
        downstream fit report.

        Parameters
        ----------
        df : pd.DataFrame
            Raw quotes; one row per option.
        strike, expiry, cp, bid, ask : str
            Column names. ``expiry`` is a year fraction; ``cp`` accepts
            'c'/'call'/'p'/'put' (case-insensitive).
        spot : float, optional
            Underlying spot, used only for the forward fallback.
        rate : float or callable, default 0.0
            Continuously compounded rate: flat float or T -> r(T).
        dividend_yield : float or callable, default 0.0
            Continuous dividend yield for the forward fallback only
            (put-call-parity forwards embed dividends already).
        mode : str, default "warn"
            Failure handling. ``"strict"``: the first bad input raises
            with its location (invalid quote rows, an expiry that
            cannot form a forward, a mid quote whose implied vol will
            not invert). ``"warn"``: problems are logged and recorded.
            ``"lenient"``: problems are filtered silently but still
            recorded. All modes record counts on :attr:`rejections`,
            and :meth:`fit` carries them onto the surface's fit report
            -- nothing disappears without a count. Unrecognized ``cp``
            values raise in every mode (schema error, not bad data).

        Returns
        -------
        OptionChain
        """
        validate_mode(mode)
        data = pd.DataFrame({
            "strike": pd.to_numeric(df[strike], errors="coerce"),
            "maturity": pd.to_numeric(df[expiry], errors="coerce"),
            "cp": df[cp].astype(str).str.lower().str[0],
            "bid": pd.to_numeric(df[bid], errors="coerce"),
            "ask": pd.to_numeric(df[ask], errors="coerce"),
        })
        bad_cp = ~data["cp"].isin(["c", "p"])
        if bad_cp.any():
            raise ValueError(
                f"unrecognized cp values: {sorted(df.loc[bad_cp, cp].astype(str).unique())}"
            )
        valid = (
            data["strike"].gt(0) & data["maturity"].gt(0)
            & data["bid"].ge(0) & data["ask"].gt(0)
            & data["ask"].ge(data["bid"])
        )
        n_dropped = int((~valid).sum())
        if n_dropped:
            if mode == "strict":
                bad_rows = df.index[~valid.to_numpy()].tolist()
                raise ValueError(
                    f"OptionChain(mode='strict'): {n_dropped} invalid quote "
                    f"rows (non-positive strike/expiry/ask, or crossed "
                    f"bid > ask); first offenders at input rows "
                    f"{bad_rows[:5]}"
                )
            if mode == "warn":
                logger.warning(f"OptionChain: dropped {n_dropped} invalid quote rows")
        data = data[valid].copy()
        if data.empty:
            raise ValueError("OptionChain: no valid quotes after cleaning")
        data["mid"] = 0.5 * (data["bid"] + data["ask"])

        rows = []
        skipped_expiries = []
        for T, g in sorted(data.groupby("maturity"), key=lambda item: item[0]):
            T = float(T)
            r_T = float(_rate_at(rate, T))
            calls = g[g["cp"] == "c"].groupby("strike")[["mid", "bid", "ask"]].mean()
            puts = g[g["cp"] == "p"].groupby("strike")[["mid", "bid", "ask"]].mean()
            pairs = calls.join(puts, how="inner", lsuffix="_c", rsuffix="_p")
            if len(pairs):
                fwd = calculate_implied_forward(
                    spot=pd.Series(np.full(len(pairs), spot if spot else 1.0)),
                    tte=pd.Series(np.full(len(pairs), T)),
                    r=rate,
                    strike=pd.Series(pairs.index.to_numpy()),
                    call_mid=pd.Series(pairs["mid_c"].to_numpy()),
                    put_mid=pd.Series(pairs["mid_p"].to_numpy()),
                )
                F = float(np.nanmedian(fwd))
            elif spot is not None:
                q_T = float(_rate_at(dividend_yield, T))
                F = float(spot) * float(np.exp((r_T - q_T) * T))
            else:
                if mode == "strict":
                    raise ValueError(
                        f"OptionChain(mode='strict'): expiry T={T:g} has no "
                        "put-call pairs and no spot for a forward fallback"
                    )
                if mode == "warn":
                    logger.warning(
                        f"OptionChain: expiry T={T:g} has no put-call pairs and no "
                        "spot for a forward fallback; skipping"
                    )
                skipped_expiries.append(T)
                continue
            if not np.isfinite(F) or F <= 0:
                if mode == "strict":
                    raise ValueError(
                        f"OptionChain(mode='strict'): expiry T={T:g} implied "
                        f"forward invalid ({F!r})"
                    )
                if mode == "warn":
                    logger.warning(f"OptionChain: expiry T={T:g} implied forward invalid; skipping")
                skipped_expiries.append(T)
                continue

            merged = calls.join(puts, how="outer", lsuffix="_c", rsuffix="_p")
            for K, row in merged.iterrows():
                K = float(K)
                flag = "c" if K >= F else "p"
                mid = choose_leg(K, F, row.get("mid_c", np.nan), row.get("mid_p", np.nan))
                if not np.isfinite(row.get(f"mid_{flag}", np.nan)):
                    # choose_leg fell back to the ITM leg; the inversion
                    # flag must follow it, or the ITM price reads as an
                    # absurd-but-finite OTM vol instead of failing.
                    flag = "p" if flag == "c" else "c"
                bid_px = row.get(f"bid_{flag}", np.nan)
                ask_px = row.get(f"ask_{flag}", np.nan)
                iv = _invert_iv(mid, F, K, r_T, T, flag)
                if not np.isfinite(iv) and mode == "strict":
                    raise ValueError(
                        f"OptionChain(mode='strict'): mid quote at strike "
                        f"{K:g}, expiry T={T:g} does not invert to an "
                        f"implied vol (mid={mid!r}, forward={F:g})"
                    )
                rows.append({
                    "strike": K,
                    "maturity": T,
                    "implied_forward": F,
                    "iv": iv,
                    "iv_bid": _invert_iv(bid_px, F, K, r_T, T, flag),
                    "iv_ask": _invert_iv(ask_px, F, K, r_T, T, flag),
                })
        if not rows:
            raise ValueError("OptionChain: no expiry produced a usable slice")
        panel = pd.DataFrame(rows)
        return cls(panel, rate=rate, rejections={
            "rejected_quotes": n_dropped,
            "failed_inversions": int(panel["iv"].isna().sum()),
            "skipped_expiries": skipped_expiries,
        })

    def fit(
        self,
        model: Union[str, Parametrization] = "svi",
        enforce_calendar: bool = False,
        arbitrage_condition: ArbitrageFreedom = ArbitrageFreedom.QUASI,
        r: Optional[float] = None,
        mode: str = "warn",
        **model_kwargs,
    ) -> VolSurface:
        """Calibrate a surface from the chain in one call.

        Routes to :func:`calibrate_surface` when ``enforce_calendar`` is
        set, else :meth:`VolSurface.fit`. ``r`` sets the surface's flat
        pricing rate; when omitted it defaults to the chain's rate if
        that is a flat float, else 0.0 (a callable term structure has no
        flat representation on the surface yet). ``mode`` sets the
        failure handling for the fit (see :meth:`from_dataframe`); the
        chain's ingestion accounting is carried onto the returned
        surface's fit report.
        """
        from dataclasses import replace

        validate_mode(mode)
        if r is None:
            r = self.rate if isinstance(self.rate, (int, float)) else 0.0
        if enforce_calendar:
            surface = calibrate_surface(
                self._panel, model=model, enforce_calendar=True,
                arbitrage_condition=arbitrage_condition, r=float(r),
                mode=mode, **model_kwargs,
            )
        else:
            surface = VolSurface.fit(
                self._panel, model=model,
                arbitrage_condition=arbitrage_condition, r=float(r),
                mode=mode, **model_kwargs,
            )
        if surface.fit_report is not None:
            surface.fit_report = replace(
                surface.fit_report,
                n_rejected_quotes=int(self.rejections["rejected_quotes"]),
                n_failed_inversions=int(self.rejections["failed_inversions"]),
                n_skipped_expiries=len(self.rejections["skipped_expiries"]),
            )
        return surface
