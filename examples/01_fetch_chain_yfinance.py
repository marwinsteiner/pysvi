"""Fetch a real SPY option chain snapshot from Yahoo Finance.

This script builds the raw-data snapshot that every other example runs
on. It is the only script that touches the network; everything
downstream reads the files it writes, so the whole pipeline is
reproducible and free of lookahead by construction.

Building a contemporaneous set without lookahead
------------------------------------------------
A volatility surface is a picture of the market at one instant. Mixing
quotes observed at different times (or quotes you could only have known
later) produces phantom arbitrage and unstable fits. The discipline:

1. Record a single snapshot timestamp BEFORE fetching anything.
2. Fetch every input you will ever use -- option quotes, spot, rates --
   in one pass, and stamp them all with that timestamp.
3. Compute time-to-expiry against the snapshot timestamp, never
   against "now" at analysis time.
4. Filter stale quotes using only information available at the
   snapshot: a quote whose last trade is days old was already stale
   when you fetched it (Yahoo keeps showing the last print).
5. Persist the raw snapshot to disk. Downstream analysis reads the
   file; it never re-fetches, so it can never accidentally peek at
   newer data.

Yahoo's free feed is delayed ~15 minutes and bid/ask can be crossed or
empty outside regular trading hours; the snapshot records what was
knowable at fetch time, which is exactly what a backtest would have had.

Usage::

    uv run --with yfinance --with interest-rate-models examples/01_fetch_chain_yfinance.py

Outputs ``examples/data/spy_chain_<UTCstamp>.csv`` (raw quotes) plus a
``.meta.json`` sidecar (snapshot timestamp, spot, rates), and refreshes
the ``spy_chain_latest.*`` copies the other examples pick up by default.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import interest_rate_models as irm
import numpy as np
import pandas as pd
import yfinance as yf

from pysvi.calibration import parse_ticker_info

DATA_DIR = Path(__file__).parent / "data"
UNDERLYING = "SPY"
N_EXPIRIES = 6            # spread across the listed curve
MONEYNESS_BAND = 0.25     # keep strikes within +/- 25% of spot
MAX_STALE_DAYS = 3        # drop quotes whose last trade predates this
MIN_PRICE = 0.10          # drop penny quotes (tick-size noise, not vol)


def annual_to_continuous(y_pct: float) -> float:
    """Yahoo yield indices quote annualized percent; convert to a
    continuously compounded rate: r = ln(1 + y)."""
    import math
    return math.log(1.0 + y_pct / 100.0)


def main() -> None:
    # 1. One timestamp, recorded before any request goes out.
    snapshot_ts = datetime.now(timezone.utc)
    stamp = snapshot_ts.strftime("%Y%m%dT%H%M%SZ")

    # 2. Fetch everything in one pass: spot, rates, then the chain.
    ticker = yf.Ticker(UNDERLYING)
    spot = float(ticker.fast_info.last_price)

    # Treasury pillars, all at the snapshot: ^IRX = 13-week bill,
    # ^FVX = 5y, ^TNX = 10y, ^TYX = 30y (annualized %). The curve
    # between and beyond the pillars is fitted below with the
    # interest-rate-models package (PyPI: interest-rate-models) --
    # log-linear discount-factor interpolation via DiscountCurve --
    # and persisted densely so the offline examples need neither the
    # network nor the extra dependency.
    tenors = {"^IRX": 0.25, "^FVX": 5.0, "^TNX": 10.0, "^TYX": 30.0}
    pillar_times = np.array(sorted(tenors.values()))
    pillar_rates = np.array([
        annual_to_continuous(float(yf.Ticker(sym).fast_info.last_price))
        for sym, _ in sorted(tenors.items(), key=lambda kv: kv[1])
    ])

    curve = irm.DiscountCurve.from_zero_rates(pillar_times, pillar_rates)
    r_13w = float(curve.zero_rate(0.25))
    # Dense fitted zero curve for the offline scripts (no irm needed
    # downstream: term_structure() in _snapshot.py just interpolates it).
    grid_times = np.geomspace(0.02, 30.0, 60)
    grid_rates = np.array([float(curve.zero_rate(t)) for t in grid_times])

    # The same curve also feeds interest-rate-models' model layer --
    # e.g. a Vasicek equilibrium fit to today's pillars:
    vasicek = irm.get_model("vasicek", kappa=0.5, theta=0.04, sigma=0.01)
    vas = vasicek.calibrate(curve, r0=r_13w)
    print("Vasicek fit to today's curve: "
          + ", ".join(f"{kk}={vv:.4f}" for kk, vv in vas.items()))

    all_expiries = ticker.options
    if not all_expiries:
        raise SystemExit("Yahoo returned no listed expiries for " + UNDERLYING)
    # Skip expiries inside a week (tiny T makes IVs numerically junky),
    # then spread the picks across the listed curve (front to back).
    usable = [
        e for e in all_expiries
        if (pd.Timestamp(e, tz="UTC") - snapshot_ts).days >= 7
    ]
    step = max(len(usable) // N_EXPIRIES, 1)
    expiries = list(usable[::step][:N_EXPIRIES])

    rows = []
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        for cp, quotes in (("c", chain.calls), ("p", chain.puts)):
            for _, q in quotes.iterrows():
                rows.append({
                    "contractSymbol": q["contractSymbol"],
                    "expiry_date": expiry,
                    "strike": float(q["strike"]),
                    "cp": cp,
                    "bid": float(q["bid"]),
                    "ask": float(q["ask"]),
                    "lastPrice": float(q["lastPrice"]),
                    "lastTradeDate": str(q["lastTradeDate"]),
                    "volume": float(q["volume"]) if pd.notna(q["volume"]) else 0.0,
                    "openInterest": (
                        float(q["openInterest"]) if pd.notna(q["openInterest"]) else 0.0
                    ),
                    "yahoo_iv": float(q["impliedVolatility"]),
                })
    df = pd.DataFrame(rows)
    n_raw = len(df)

    # Yahoo contract symbols are OCC format; pysvi ships a parser for
    # OCC-named files -- it works on the symbols directly too:
    parsed = parse_ticker_info(Path(df["contractSymbol"].iloc[0]))
    print(f"parse_ticker_info({df['contractSymbol'].iloc[0]!r}) -> {parsed}")

    # 3/4. Snapshot-time filters only. Everything here was knowable at
    # snapshot_ts: the moneyness band uses the snapshot spot, and the
    # staleness test compares last trade time to the snapshot time.
    df = df[(df["strike"] >= spot * (1 - MONEYNESS_BAND))
            & (df["strike"] <= spot * (1 + MONEYNESS_BAND))]
    last_trade = pd.to_datetime(df["lastTradeDate"], utc=True, format="mixed")
    fresh = (snapshot_ts - last_trade).dt.total_seconds() <= MAX_STALE_DAYS * 86400
    df = df[fresh]

    # Outside regular trading hours Yahoo zeroes the book: bid = ask =
    # 0 and only the last print survives. When most of the book is
    # empty, fall back to last-traded prices (bid = ask = lastPrice) --
    # on a weekend that IS the best contemporaneous information. Two
    # honest caveats, recorded in the metadata: last prints are
    # asynchronous across strikes (each traded at a different moment of
    # the final session), and the bid/ask columns then carry no spread,
    # so downstream iv_bid/iv_ask collapse onto the mid. For production
    # surfaces, snapshot during market hours and get the live book.
    two_sided = (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])
    if two_sided.mean() >= 0.2:
        quote_source = "live bid/ask"
        df = df[two_sided]
    else:
        quote_source = "last trade (closed book: bid = ask = lastPrice)"
        df = df[df["lastPrice"] > 0].copy()
        df["bid"] = df["lastPrice"]
        df["ask"] = df["lastPrice"]
    # Penny prints on far wings are tick-size lottery tickets, not vol
    # information: a $0.02 print inverted at short maturity produces an
    # absurd implied vol and will dominate any least-squares fit.
    df = df[0.5 * (df["bid"] + df["ask"]) >= MIN_PRICE]
    print(f"quote source: {quote_source}")
    print(f"kept {len(df)}/{n_raw} quotes after moneyness/staleness/quote filters")

    # 5. Persist the raw snapshot + sidecar; refresh the *_latest copies.
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"spy_chain_{stamp}.csv"
    meta_path = csv_path.with_suffix("").with_suffix(".meta.json")
    df.to_csv(csv_path, index=False)
    meta = {
        "underlying": UNDERLYING,
        "snapshot_utc": snapshot_ts.isoformat(),
        "spot": spot,
        "r_13w_cc": r_13w,
        "curve_pillars": {str(t): float(r)
                          for t, r in zip(pillar_times, pillar_rates)},
        "zero_curve": {"times": grid_times.tolist(),
                       "rates": grid_rates.tolist(),
                       "fitted_with": "interest-rate-models DiscountCurve"},
        "expiries": expiries,
        "moneyness_band": MONEYNESS_BAND,
        "max_stale_days": MAX_STALE_DAYS,
        "quote_source": quote_source,
        "source": "yfinance (delayed Yahoo Finance feed)",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    shutil.copy(csv_path, DATA_DIR / "spy_chain_latest.csv")
    shutil.copy(meta_path, DATA_DIR / "spy_chain_latest.meta.json")

    print(f"spot={spot:.2f}  zero curve: "
          + "  ".join(f"{tt:g}y={rr:.3%}" for tt, rr in
                      zip(pillar_times, pillar_rates)))
    print(f"wrote {csv_path.name} ({len(df)} quotes, {len(expiries)} expiries)")
    print("downstream examples read spy_chain_latest.csv -- no further network access")


if __name__ == "__main__":
    main()
