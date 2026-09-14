"""Every parametrization on one real slice, plus the analytics API.

Calibrates all seven models in the library to the same real SPY expiry
(from the 01 snapshot), then tours the per-model analytics: fitted-IV
columns via ``apply_slice``, the module-level total-variance functions,
the raw/natural SVI bijection, derivatives, the risk-neutral density,
asymptotic wing slopes, and the single-slice arbitrage report.

Model cheat sheet (details: https://pysvi.readthedocs.io/en/latest/models/):

* ``svi``      raw SVI (Gatheral 2004), 5 params, the workhorse
* ``natural``  natural SVI (Gatheral-Jacquier 2014), bijective with raw
* ``ssvi``     surface SVI -- needs the slice ATM total variance ``theta``
* ``essvi``    extended SSVI, rho a function of theta -- ``theta`` (+
  optional ``theta_ref``)
* ``jw``       jump-wings: trader-facing params -- needs ``T``
* ``dsvi``     DirectSVI (Schadner): closed-form conic fit, no optimizer
* ``sabr``     SABR (Hagan et al. 2002) -- needs ``T`` and ``F``,
  optional fixed ``beta``

Usage::

    uv run examples/03_single_slice_models.py
"""

import numpy as np
import pandas as pd

from pysvi import (
    SVI, ArbitrageFreedom, LEE_BOUND,
    apply_slice, calibrate_slice, check_slice_arbitrage, get_model,
    natural_to_raw, raw_to_natural,
    directsvi_total_variance, essvi_total_variance, jw_total_variance,
    natural_total_variance, sabr_implied_vol, sabr_total_variance,
    ssvi_total_variance, svi_total_variance,
)
from pysvi.calibration import choose_leg, compute_ivs_vectorized

from _snapshot import load_snapshot


def build_slice(df, meta, expiry_index: int = 2) -> pd.DataFrame:
    """One expiry in the calibrate_slice schema (same steps as example 02)."""
    T = float(sorted(df["expiry"].unique())[expiry_index])
    g = df[df["expiry"] == T].copy()
    g["mid"] = 0.5 * (g["bid"] + g["ask"])
    calls = g[g["cp"] == "c"].groupby("strike")["mid"].mean()
    puts = g[g["cp"] == "p"].groupby("strike")["mid"].mean()
    both = calls.index.intersection(puts.index)
    F = float(np.nanmedian(
        both.to_numpy()
        + np.exp(meta["r_13w_cc"] * T)
        * (calls.loc[both].to_numpy() - puts.loc[both].to_numpy())
    ))
    strikes = calls.index.union(puts.index).to_numpy(dtype=float)
    legs, flags = [], []
    for K in strikes:
        c_mid, p_mid = calls.get(K, np.nan), puts.get(K, np.nan)
        legs.append(choose_leg(K, F, c_mid, p_mid))
        flag = "c" if K >= F else "p"          # flag follows the leg used
        if not np.isfinite(c_mid if flag == "c" else p_mid):
            flag = "p" if flag == "c" else "c"
        flags.append(flag)
    ivs = compute_ivs_vectorized(
        prices=np.asarray(legs), spots=np.full(len(strikes), meta["spot"]),
        strikes=strikes, ttes=np.full(len(strikes), T),
        r=meta["r_13w_cc"], flags=np.asarray(flags),
    )
    return pd.DataFrame({
        "strike": strikes, "iv": ivs, "maturity": T, "implied_forward": F,
    }).dropna()


def main() -> None:
    df, meta = load_snapshot()
    df_slice = build_slice(df, meta)
    T = float(df_slice["maturity"].iloc[0])
    F = float(df_slice["implied_forward"].iloc[0])
    theta = float(np.nanmin(df_slice["iv"] ** 2 * df_slice["maturity"]))
    print(f"slice: T={T:.3f}y  F={F:.2f}  {len(df_slice)} quotes  "
          f"theta(ATM total var)={theta:.5f}\n")

    # ── Calibrate all seven models to the same real slice ────────────
    # get_model takes the lowercase name and an ArbitrageFreedom flag;
    # per-model extras go through calibrate_slice as keyword arguments.
    # Initialization matters on real data and is worth MEASURING per
    # model: here multi_start beats the default start by 3-5x for
    # svi/ssvi/essvi/sabr, while jw does better from its data-driven
    # default on this snapshot. dsvi is closed-form (no controls).
    runs = [
        ("svi", {"initialization": "multi_start"}),
        ("natural", {"initialization": "multi_start"}),
        ("ssvi", {"theta": theta, "initialization": "multi_start"}),
        ("essvi", {"theta": theta, "theta_ref": theta,
                   "initialization": "multi_start"}),
        ("jw", {"T": T}),
        ("dsvi", {}),
        ("sabr", {"T": T, "F": F, "beta": 0.5,
                  "initialization": "multi_start"}),
    ]
    fits = {}
    for name, kwargs in runs:
        model = get_model(name, ArbitrageFreedom.NO_BUTTERFLY)
        params = calibrate_slice(df_slice, model, **kwargs)
        if params is None:
            print(f"{name:>8}: calibration failed")
            continue
        fitted = apply_slice(
            df_slice, params, model,
            maturity_col="maturity", strike_col="strike", iv_col="iv",
            fitted_col="fitted_iv", residual_col="residual_iv",
        )
        rmse = float(np.sqrt(np.mean(fitted["residual_iv"] ** 2)))
        fits[name] = (model, params)
        print(f"{name:>8}: IV RMSE {rmse * 1e4:6.1f} bp   "
              f"params {['%s=%.4g' % kv for kv in params.items() if kv[0] != 'forward']}")

    # ── Module-level total-variance functions ────────────────────────
    # Each parametrization is also exposed as a plain function of
    # (k, *params) -- handy for plotting or embedding without objects.
    k = np.linspace(-0.15, 0.15, 7)
    p = fits["svi"][1]
    w_fn = svi_total_variance(k, p["a"], p["b"], p["rho"], p["m"], p["sigma"])
    w_obj = fits["svi"][0].total_variance(k, p)
    assert np.allclose(w_fn, w_obj)
    print("\nsvi_total_variance(k, a, b, rho, m, sigma) == SVI().total_variance")
    print("also available: ssvi_/essvi_/natural_/jw_/directsvi_total_variance,")
    print("sabr_total_variance and sabr_implied_vol", end="")
    if "sabr" in fits:
        ps = fits["sabr"][1]
        atm_sabr = sabr_implied_vol(
            np.array([0.0]), ps["alpha"], ps["beta"], ps["rho"], ps["nu"], F=F, T=T,
        )[0]
        print(f" (SABR ATM vol: {atm_sabr:.2%})")
    else:
        print()
    _ = (ssvi_total_variance, essvi_total_variance, natural_total_variance,
         jw_total_variance, directsvi_total_variance, sabr_total_variance)

    # ── Raw <-> natural SVI bijection ────────────────────────────────
    nat = raw_to_natural(**{kk: p[kk] for kk in ("a", "b", "rho", "m", "sigma")})
    back = natural_to_raw(**nat)
    print(f"\nraw_to_natural(raw SVI fit) -> {({kk: round(vv, 4) for kk, vv in nat.items()})}")
    assert all(np.isclose(back[kk], p[kk]) for kk in back)
    print("natural_to_raw(raw_to_natural(p)) round-trips exactly")

    # ── Derivatives, density, wing slopes ────────────────────────────
    model, p = fits["svi"]
    w, dw, d2w = model.derivatives(k, p)          # (w, w', w'') in one call
    assert np.allclose(dw, model.dw_dk(k, p))     # or individually
    assert np.allclose(d2w, model.d2w_dk2(k, p))
    g = model.density(k, p)                       # risk-neutral density g(k)
    lo, hi = model.wing_slopes(p)                 # asymptotic slopes b(1 -+ rho)
    print(f"\nATM skew dw/dk = {model.dw_dk(np.array([0.0]), p)[0]:+.5f}, "
          f"curvature d2w/dk2 = {model.d2w_dk2(np.array([0.0]), p)[0]:+.5f}")
    print(f"density g(k) >= {g.min():.4f} on the quoted range "
          f"(negative would mean butterfly arbitrage)")
    print(f"wing slopes ({lo:.4f}, {hi:.4f}); Lee's moment bound caps both at "
          f"{LEE_BOUND}")
    # Models without analytic derivatives (SABR, DirectSVI) fall back to
    # central finite differences with step model.fd_step (default 1e-5),
    # tunable per instance:
    fits["dsvi"][0].fd_step = 1e-4

    # ── Single-slice arbitrage report ────────────────────────────────
    # k_data mirrors the quoted range into the check grid; tol sets the
    # violation tolerance; the report prints as a formatted block.
    report = check_slice_arbitrage(
        model, p,
        maturity=T,
        k_min=-1.0, k_max=1.0, n_grid=801,
        tol=1e-8,
        k_data=np.log(df_slice["strike"] / df_slice["implied_forward"]).to_numpy(),
    )
    print(f"\ncheck_slice_arbitrage: ok={report.ok} "
          f"(butterfly_free={report.butterfly_free}, "
          f"lee_free={report.lee_free})")
    print(report)


if __name__ == "__main__":
    main()
