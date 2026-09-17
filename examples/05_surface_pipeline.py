"""The full pipeline: raw SPY quotes to a priced, verified, saved surface.

Everything the earlier examples did by hand, in two calls -- then a
tour of the entire ``VolSurface`` API on the result: evaluation,
maturity interpolation, diagnostics, Black-76 pricing and Greeks, and
serialization.

Usage::

    uv run examples/05_surface_pipeline.py
"""

from pathlib import Path

import numpy as np

from pysvi import (
    ArbitrageFreedom, OptionChain, VolSurface, calibrate_surface, get_model,
)

from _snapshot import load_snapshot, term_structure

OUT = Path(__file__).parent / "data"


def main() -> None:
    df, meta = load_snapshot()
    spot, r_flat = meta["spot"], meta["r_13w_cc"]
    curve = term_structure(meta)             # irm.DiscountCurve

    # ── OptionChain: raw quotes in, calibration panel out ────────────
    # Column names are configurable (here the snapshot already uses
    # different names than the defaults for expiry). rate accepts a
    # flat float or a callable T -> r(T); spot and dividend_yield are
    # only the forward FALLBACK for expiries missing put-call pairs --
    # whenever both legs quote, the forward comes from put-call parity
    # and embeds dividends automatically.
    chain = OptionChain.from_dataframe(
        df,
        strike="strike",
        expiry="expiry",          # year fraction (computed in _snapshot)
        cp="cp",                  # accepts c/call/p/put, any case
        bid="bid",
        ask="ask",
        spot=spot,
        rate=curve,               # the irm curve object, directly
        dividend_yield=0.012,     # used only by the spot fallback
    )
    panel = chain.panel           # calibrate_slice schema + iv_bid/iv_ask
    print(f"chain: {len(panel)} options, maturities "
          f"{[round(T, 3) for T in chain.maturities]}")
    print(f"panel columns: {list(panel.columns)}")

    # ── Failure semantics: strict / warn / lenient ───────────────────
    # Nothing disappears silently: every mode records rejected quotes,
    # failed inversions, and skipped expiries on chain.rejections, and
    # fit() carries the counts onto the surface's fit report. strict
    # raises on the first bad input with its location; lenient filters
    # without logging. Demonstrate strict on a deliberately bad quote:
    print(f"ingestion accounting: {chain.rejections}")
    corrupted = df.copy()
    corrupted.loc[corrupted.index[0], ["bid", "ask"]] = [5.0, 4.0]  # crossed
    try:
        OptionChain.from_dataframe(corrupted, strike="strike", expiry="expiry",
                                   cp="cp", bid="bid", ask="ask",
                                   rate=r_flat, mode="strict")
    except ValueError as err:
        print(f"mode='strict' raised as designed: {str(err)[:70]}...")

    # ── Route 1: chain.fit -- independent slices ─────────────────────
    surface = chain.fit(
        model="svi",
        enforce_calendar=False,
        arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY,
        r=r_flat,                 # flat pricing rate for the surface
        initialization="multi_start",
    )
    rmses = [s.iv_rmse for s in surface.fit_report.slices if s.iv_rmse is not None]
    print(f"\nchain.fit(svi): {surface.fit_report.n_ok} slices ok, "
          f"max RMSE {max(rmses) * 1e4:.1f} bp")

    # ── Route 2: calendar-aware joint calibration ────────────────────
    # calibrate_surface chains each slice's total variance into the
    # next slice's calendar penalty; for eSSVI the shape parameters are
    # fitted jointly across expiries. Works from the chain's panel or
    # any DataFrame in the same schema.
    surface = calibrate_surface(
        panel,
        model="essvi",
        enforce_calendar=True,
        arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY,
        r=r_flat,
        interp_method="theta",    # parametric maturity interpolation
    )

    # ── Diagnose before trusting ─────────────────────────────────────
    # diagnose() = fit report + arbitrage checks in one block, run on
    # the quoted moneyness range by default. check_arbitrage exposes
    # the grid controls directly.
    diag = surface.diagnose()
    print(f"\ndiagnose(): ok={diag.ok}")
    print(diag)
    # Verdicts are grid-dependent: a marginal violation just outside
    # the quoted range can appear on one grid and not another. Choose
    # the region you actually need certified.
    report = surface.check_arbitrage(k_min=-0.5, k_max=0.5, n_grid=401, tol=1e-8)
    print(f"explicit-grid check_arbitrage: ok={report.ok}, "
          f"calendar_free={report.calendar_free}")

    # ── Evaluation API ───────────────────────────────────────────────
    T1, T2 = float(chain.maturities[0]), float(chain.maturities[-1])
    T_mid = 0.5 * (T1 + T2)       # NOT a quoted expiry: interpolated
    K = np.array([spot * 0.9, spot, spot * 1.1])
    print(f"\nforward({T_mid:.3f}) = {surface.forward(T_mid):.2f} "
          f"(log-linear between fitted slices)")
    print(f"iv(K, {T_mid:.3f})           = {np.round(surface.iv(K, T_mid), 4)}")
    print(f"total_variance(k=0)      = {surface.total_variance(0.0, T_mid):.5f}")
    print(f"atm_vol / skew / curvature at T={T1:.3f}: "
          f"{surface.atm_vol(T1):.4f} / {surface.skew(T1):+.4f} / "
          f"{surface.curvature(T1):+.4f}")
    print(f"params({T1:.3f}) keys    = {sorted(surface.params(T1))}")
    # slice_at works between expiries because interp_method='theta'
    # gives a parametric slice there (the default 'total_variance'
    # blend evaluates anywhere but has no parameter representation):
    print(f"slice_at({T_mid:.3f}) theta = {surface.slice_at(T_mid)['theta']:.5f}")

    # ── Black-76 pricing and Greeks ──────────────────────────────────
    # Sticky-strike: each Greek holds the strike's implied vol fixed.
    # delta is the forward delta e^{-rT} N(d1); vega is per unit vol;
    # theta is per year of calendar time.
    for cp in ("call", "put"):
        print(f"{cp:>5}: price {np.round(surface.price(K, T_mid, cp=cp), 3)}  "
              f"delta {np.round(surface.delta(K, T_mid, cp=cp), 3)}  "
              f"theta {np.round(surface.theta(K, T_mid, cp=cp), 3)}")
    print(f"gamma {np.round(surface.gamma(K, T_mid), 5)}  "
          f"vega {np.round(surface.vega(K, T_mid), 3)} (cp-independent)")

    # ── Serialization ────────────────────────────────────────────────
    # Versioned JSON; load() reproduces evaluation bitwise, fit report
    # and provenance included. Calibrate once, distribute the file.
    path = OUT / "spy_surface.json"
    surface.save(path)
    loaded = VolSurface.load(path)
    assert np.array_equal(loaded.iv(K, T_mid), surface.iv(K, T_mid))
    print(f"\nsaved -> {path.name}; load() round-trips evaluation exactly "
          f"(pysvi {loaded.fit_report.pysvi_version})")

    # ── Differentiability: C1 maturity interpolation ─────────────────
    # The default total-variance blend is continuous but its maturity
    # derivative jumps at every fitted slice (regularity "C0"): fine
    # for prices and IVs, wrong for anything consuming dw/dT (Dupire
    # local vol, forward variance). interp_method="monotone_cubic" is
    # a shape-preserving cubic in T -- C1, exact at fitted maturities,
    # calendar-monotone -- and unlocks the dw_dT accessor.
    c1 = VolSurface.fit(
        panel, model="svi", r=r_flat,
        interp_method="monotone_cubic", initialization="multi_start",
    )
    print(f"\nregularity: default={surface.regularity}, "
          f"monotone_cubic={c1.regularity}")
    print(f"dw_dT(ATM, T={T_mid:.3f}) = {c1.dw_dT(0.0, T_mid):.5f} "
          "(the Dupire numerator)")

    # ── Direct construction ──────────────────────────────────────────
    # VolSurface is just (model, {maturity: params}) -- params from any
    # source work as long as each dict carries 'forward'. Useful for
    # loading externally calibrated surfaces.
    direct = VolSurface(
        get_model("essvi"),
        {T: surface.params(T) for T in surface.maturities},
        r=r_flat,
        interp_method="total_variance",
    )
    assert np.allclose(direct.iv(K, T1), surface.iv(K, T1))
    print("direct VolSurface(model, slices) construction matches the fit")


if __name__ == "__main__":
    main()
