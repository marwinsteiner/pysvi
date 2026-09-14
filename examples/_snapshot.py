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
    """Continuously compounded zero rate r(T) from the two snapshot tenors.

    Linear interpolation between the 13-week and 10-year points, flat
    beyond -- a toy curve, but built entirely from snapshot-time data.
    """
    tenors = np.array([0.25, 10.0])
    rates = np.array([meta["r_13w_cc"], meta["r_10y_cc"]])
    return lambda T: float(np.interp(T, tenors, rates))
