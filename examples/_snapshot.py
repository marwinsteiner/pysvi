"""Shared loader for the snapshot written by 01_fetch_chain_yfinance.py.

Time-to-expiry is computed against the recorded snapshot timestamp --
never against the wall clock -- so re-running an example tomorrow gives
bitwise the same numbers. Equity options expire at the close; we pin
expiry to 21:00 UTC (4pm New York, ignoring the DST hour, immaterial at
this horizon) and use an ACT/365 year fraction.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_snapshot(name: str = "spy_chain_latest"):
    """Return (quotes DataFrame with a 'expiry' year-fraction column, meta dict)."""
    csv_path = DATA_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found -- run 01_fetch_chain_yfinance.py first "
            "(uv run --with yfinance examples/01_fetch_chain_yfinance.py)"
        )
    meta = json.loads(csv_path.with_suffix("").with_suffix(".meta.json").read_text())
    df = pd.read_csv(csv_path)

    snapshot_ts = datetime.fromisoformat(meta["snapshot_utc"])
    expiry_close = pd.to_datetime(df["expiry_date"]).dt.tz_localize(timezone.utc) \
        + pd.Timedelta(hours=21)
    df["expiry"] = (expiry_close - snapshot_ts).dt.total_seconds() / (365.0 * 86400.0)
    df = df[df["expiry"] > 0].reset_index(drop=True)
    return df, meta


def term_structure(meta):
    """The snapshot's Treasury curve as an ``irm.DiscountCurve``.

    Script 01 fits a real four-pillar curve (13w/5y/10y/30y) with
    interest-rate-models -- a core dependency of svi-py -- and persists
    it densely in the metadata; this rebuilds the same DiscountCurve
    from the stored grid, so every reconstruction uses only data
    observable at the snapshot timestamp. The returned object can be
    passed directly wherever svi-py takes a rate (its ``zero_rate`` is
    the r(T) accessor).
    """
    import interest_rate_models as irm

    zc = meta["zero_curve"]
    return irm.DiscountCurve.from_zero_rates(
        np.asarray(zc["times"], dtype=float),
        np.asarray(zc["rates"], dtype=float),
    )
