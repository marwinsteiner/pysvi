"""Trading workflows on the fitted surface: stability, sensitivities,
events, conventions, and economic arbitrage.

The v1.2.0 layer, demonstrated on the real SPY snapshot:

* prior-anchored recalibration -- keep parameter paths stable across
  snapshots instead of basin-hopping between equivalent fits
* quote-to-surface Jacobians -- "this quote moves 1 vol point: what
  does the smile do", without recalibrating
* variance events -- a known earnings/FOMC jump between expiries,
  interpolated as a jump instead of smeared
* MarketContext -- real expiry dates plus one coherent day-count and
  numeraire convention
* evaluation status and economic arbitrage classification -- is a
  number market information or model extrapolation, and is a violation
  a trade or an artifact

Usage::

    uv run examples/06_trading_workflows.py
"""

import numpy as np

from pysvi import (
    MarketContext, OptionChain, VarianceEvent, VolSurface,
    classify_arbitrage, implied_event_variances,
    iv_surface_sensitivity, quote_sensitivity,
)
from pysvi.calibration import prepare_slice

from _snapshot import load_snapshot, term_structure


def main() -> None:
    df, meta = load_snapshot()
    spot, r_flat = meta["spot"], meta["r_13w_cc"]
    curve = term_structure(meta)

    chain = OptionChain.from_dataframe(
        df, strike="strike", expiry="expiry", cp="cp", bid="bid", ask="ask",
        spot=spot, rate=curve, dividend_yield=0.012,
    )
    panel = chain.panel
    surface = chain.fit(model="svi", r=r_flat, initialization="multi_start")

    # ── Prior-anchored recalibration ─────────────────────────────────
    # Tomorrow's quotes move a little; without anchoring the optimizer
    # may hop to an equivalent basin and your parameter time series is
    # noise. prior= warm-starts from today's fit and anchor= adds a
    # Tikhonov pull toward today's SURFACE SHAPE (not raw parameters --
    # they are not equally meaningful; see identifiability).
    rng = np.random.default_rng(1)
    bumped = panel.copy()
    bumped["iv"] = bumped["iv"] * (1 + 5e-4 * rng.standard_normal(len(bumped)))
    s_free = VolSurface.fit(bumped, model="svi", r=r_flat,
                            initialization="multi_start")
    s_anch = VolSurface.fit(bumped, model="svi", r=r_flat,
                            initialization="multi_start",
                            prior=surface, anchor=1.0)
    T0 = float(surface.maturities[2])
    names = ("a", "b", "rho", "m", "sigma")
    d_free = max(abs(s_free.params(T0)[n] - surface.params(T0)[n]) for n in names)
    d_anch = max(abs(s_anch.params(T0)[n] - surface.params(T0)[n]) for n in names)
    print(f"parameter drift on a ~5bp iv perturbation (T={T0:.3f}): "
          f"unanchored {d_free:.5f}, anchored {d_anch:.6f}")

    # ── Quote-to-surface Jacobians ───────────────────────────────────
    # Implicit-function sensitivities at the optimum: bump quote i,
    # read the whole smile's response -- no recalibration.
    g = panel[np.isclose(panel["maturity"], T0)]
    k, w, F = prepare_slice(g)
    model, params = surface.model, surface.params(T0)
    S = quote_sensitivity(model, params, k, w)
    S_iv = iv_surface_sensitivity(model, params, k, w, k, T0)
    i_atm = int(np.argmin(np.abs(k)))
    print(f"ATM quote +1 vol pt moves the ATM fit by "
          f"{S_iv[i_atm, i_atm]:.2f} vol pts (dtheta/dw shape {S.shape})")

    # ── Variance events ──────────────────────────────────────────────
    # A known event between two quoted expiries: fit the continuous
    # component, evaluate with the jump on the right side.
    T1, T2 = float(chain.maturities[2]), float(chain.maturities[3])
    event = VarianceEvent(0.5 * (T1 + T2), 0.0004, "synthetic earnings")
    s_ev = chain.fit(model="svi", r=r_flat, initialization="multi_start",
                     enforce_calendar=True, events=[event])
    before = s_ev.total_variance(0.0, event.time - 1e-4)
    after = s_ev.total_variance(0.0, event.time + 1e-4)
    rep_ev = s_ev.diagnose().arbitrage
    rep_no = chain.fit(model="svi", r=r_flat, initialization="multi_start",
                       enforce_calendar=True).diagnose().arbitrage
    # The calendar check sees the CONTINUOUS component, so the event
    # itself triggers nothing: any flag present is the same wing
    # artifact the no-event fit shows on this data (and the economic
    # classification below labels such wing findings for what they are).
    print(f"event at T={event.time:.3f}: w jumps {after - before:.5f} "
          f"(specified {event.variance}); event adds no calendar flag: "
          f"{rep_ev.calendar_free == rep_no.calendar_free}")
    for rep in implied_event_variances(panel, [event]):
        print(f"quoted ATM-w jump across the event: {rep['quoted_jump']:.5f} "
              f"(upper bound on the event variance)")

    # ── MarketContext: dates and one coherent convention ─────────────
    # The snapshot carries real expiry DATES; a MarketContext turns
    # them into year fractions under an explicit day count and supplies
    # the numeraire -- mixing it with separate rate/spot inputs raises.
    ctx = MarketContext(meta["snapshot_utc"][:10], spot=spot, rate=curve,
                        dividend_yield=0.012, day_count="ACT/365F")
    chain_ctx = OptionChain.from_dataframe(
        df.rename(columns={"expiry_date": "expiry_dt"}),
        strike="strike", expiry="expiry_dt", cp="cp", bid="bid", ask="ask",
        context=ctx,
    )
    print(f"context chain (dates + ACT/365F): maturities "
          f"{[round(float(T), 3) for T in chain_ctx.maturities]}")
    print(f"ctx.discount(1.0) = {ctx.discount(1.0):.5f}, "
          f"BUS/252 would give T(first expiry) = "
          f"{MarketContext(meta['snapshot_utc'][:10], day_count='BUS/252').year_fraction(df['expiry_date'].iloc[0]):.4f}")

    # ── Evaluation status: information vs extrapolation ──────────────
    K = np.array([spot * 0.5, spot, spot * 1.05])
    iv, status = surface.iv(K, T0, return_status=True)
    for Ki, vi, st in zip(K, iv, status):
        print(f"iv(K={Ki:7.1f}, T={T0:.3f}) = {vi:.2%}  [{st}]")

    # ── Economic arbitrage classification ────────────────────────────
    # The diagnostics report mathematical evidence; the classification
    # says what it means: outside the quoted range it is extrapolation
    # risk, inside it is executable only if a static butterfly at the
    # quoted crossed prices has negative cost.
    for c in classify_arbitrage(surface, panel=panel):
        print(c)


if __name__ == "__main__":
    main()
