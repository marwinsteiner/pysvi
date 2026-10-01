"""From raw quotes to calibration inputs: forwards, implied vols, slices.

Runs on the snapshot from 01_fetch_chain_yfinance.py and walks the
preprocessing helpers one at a time -- the manual version of what
``OptionChain`` (example 05) does in a single call:

* ``calculate_implied_forward`` -- put-call-parity forwards, with a
  flat rate and with a term-structure callable
* ``choose_leg`` -- OTM leg selection
* ``compute_ivs_vectorized`` -- Black-Scholes-Merton implied vols
* ``prepare_slice`` -- panel to (k, w, F) calibration inputs

On implied-vol inversion methods
--------------------------------
There is no closed-form inverse of the Black-Scholes price, so an
"implied vol" is always the output of some inversion algorithm:

* Textbook route: root-find the Black-76 / BSM price in sigma (Newton
  or Brent). Simple, but needs care near intrinsic value and in the
  wings.
* "Let's Be Rational" (LBR), Peter Jackel, Wilmott 2015 -- the de facto
  standard: two rational-approximation branches plus at most two
  Householder steps give full machine precision in ~ the cost of two
  price evaluations. Paper: http://www.jaeckel.org/LetsBeRational.pdf
  (journal: https://doi.org/10.1002/wilm.10395), reference code:
  https://github.com/vollib/lets_be_rational.
  This is what svi-py itself uses: ``compute_ivs_vectorized`` and
  ``OptionChain`` invert through py_vollib, whose engine is
  py_lets_be_rational -- so every IV in this pipeline is an LBR IV.
* Volfi, Wolfgang Schadner (also the author of the DirectSVI
  parametrization shipped in svi-py) -- newer work: an *explicit*
  (non-iterative) inverse via a generalized-inverse-Gaussian quantile
  representation, optionally sharpened by a single Halley step.
  Paper: "An Explicit Solution to Black-Scholes Implied Volatility"
  (https://arxiv.org/abs/2604.24480), code:
  https://github.com/wol-fi/volfi. See also fast-vollib
  (https://arxiv.org/abs/2604.27210) for vectorized LBR at scale.

Usage::

    uv run examples/02_implied_vol_and_forwards.py
"""

import numpy as np
import pandas as pd

from pysvi import calculate_implied_forward, prepare_slice
from pysvi.calibration import choose_leg, compute_ivs_vectorized

from _snapshot import load_snapshot, term_structure


def main() -> None:
    df, meta = load_snapshot()
    spot = meta["spot"]
    r_flat = meta["r_13w_cc"]
    curve = term_structure(meta)             # irm.DiscountCurve
    print(f"snapshot {meta['snapshot_utc']}  spot={spot:.2f}  "
          f"r_flat={r_flat:.4%}  r(5y)={curve.zero_rate(5.0):.4%}")

    df["mid"] = 0.5 * (df["bid"] + df["ask"])

    # ── Implied forwards from put-call parity ────────────────────────
    # C - P = e^{-rT}(F - K) at each strike quoting both legs; the
    # median across strikes is a robust forward for the expiry.
    T0 = float(sorted(df["expiry"].unique())[1])
    g = df[df["expiry"] == T0]
    calls = g[g["cp"] == "c"].set_index("strike")["mid"]
    puts = g[g["cp"] == "p"].set_index("strike")["mid"]
    both = calls.index.intersection(puts.index)

    n = len(both)
    fwd_flat = calculate_implied_forward(
        spot=pd.Series(np.full(n, spot)),
        tte=pd.Series(np.full(n, T0)),
        r=r_flat,                                # flat float rate
        strike=pd.Series(both.to_numpy()),
        call_mid=pd.Series(calls.loc[both].to_numpy()),
        put_mid=pd.Series(puts.loc[both].to_numpy()),
    )
    # The rate argument takes any rate view: an irm.DiscountCurve (as
    # here), an interest-rate model from interest_rate_models, or any
    # callable T -> r(T) -- e.g. a cubic spline over the curve pillars:
    #     from scipy.interpolate import CubicSpline
    #     r = CubicSpline(pillar_times, pillar_rates)
    fwd_curve = calculate_implied_forward(
        spot=pd.Series(np.full(n, spot)),
        tte=pd.Series(np.full(n, T0)),
        r=curve,                                 # the fitted curve itself
        strike=pd.Series(both.to_numpy()),
        call_mid=pd.Series(calls.loc[both].to_numpy()),
        put_mid=pd.Series(puts.loc[both].to_numpy()),
    )
    F = float(np.nanmedian(fwd_flat))
    print(f"T={T0:.3f}y: {n} put-call pairs, median forward {F:.2f} "
          f"(flat rate) vs {float(np.nanmedian(fwd_curve)):.2f} (curve); "
          f"spot carry check F/S-1 = {F / spot - 1:+.3%}")

    # ── OTM leg selection ────────────────────────────────────────────
    # ITM prices are mostly intrinsic value, so their extrinsic (vol)
    # content is noisy; choose_leg picks the OTM side, falling back to
    # the other leg when the preferred quote is missing.
    merged = pd.DataFrame({"call_mid": calls, "put_mid": puts}).reindex(
        calls.index.union(puts.index)
    )
    legs, flags = [], []
    for K, row in merged.iterrows():
        legs.append(choose_leg(K, F, row["call_mid"], row["put_mid"]))
        # The inversion flag must match the leg actually returned:
        # when the OTM quote is missing, choose_leg falls back to the
        # ITM leg, and inverting an ITM price with the OTM flag gives
        # an absurd-but-finite vol instead of a clean failure.
        flag = "c" if K >= F else "p"
        if not np.isfinite(row["call_mid" if flag == "c" else "put_mid"]):
            flag = "p" if flag == "c" else "c"
        flags.append(flag)
    flags = np.array(flags)
    print(f"choose_leg: {int((flags == 'p').sum())} puts, "
          f"{int((flags == 'c').sum())} calls selected")

    # ── Implied vols (LBR under the hood, see module docstring) ──────
    strikes = merged.index.to_numpy(dtype=float)
    ivs = compute_ivs_vectorized(
        prices=np.asarray(legs, dtype=float),
        spots=np.full(len(strikes), spot),
        strikes=strikes,
        ttes=np.full(len(strikes), T0),
        r=r_flat,
        q=0.0,           # SPY pays dividends; example 05 handles them
                         # properly by pricing off the implied forward
        flags=flags,
    )
    ok = np.isfinite(ivs)
    print(f"compute_ivs_vectorized: {int(ok.sum())}/{len(ivs)} inverted, "
          f"ATM ~ {ivs[np.argmin(np.abs(strikes - F))]:.2%}")

    # ── prepare_slice: panel -> (k, w, F) ────────────────────────────
    # Column names are configurable; min_points rejects slices too thin
    # to constrain a five-parameter model.
    panel = pd.DataFrame({
        "K": strikes[ok], "sigma": ivs[ok], "tte": T0, "fwd": F,
    })
    k, w, F_out = prepare_slice(
        panel,
        maturity_col="tte",
        strike_col="K",
        iv_col="sigma",
        forward_col="fwd",
        min_points=5,
    )
    print(f"prepare_slice: {len(k)} points, log-moneyness "
          f"[{k.min():+.3f}, {k.max():+.3f}], total variance "
          f"[{w.min():.5f}, {w.max():.5f}], F={F_out:.2f}")
    print("these (k, w) arrays are exactly what model.calibrate consumes "
          "-- continued in 03_single_slice_models.py")


if __name__ == "__main__":
    main()
