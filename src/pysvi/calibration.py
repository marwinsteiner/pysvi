# src/pysvi/calibration.py
"""
High-level calibration pipeline for IV surfaces from option panels.
Supports SVI, SSVI, eSSVI via models.Parametrization classes.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import re

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from py_lets_be_rational.exceptions import BelowIntrinsicException
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as bsm_iv

from .models import SVI, SSVI, ESSVI, Parametrization

warnings.filterwarnings("ignore")

_ticker_re = re.compile(r"SPY(\d{6})([CP])(\d+)")


def parse_ticker_info(path: Path):
    """Parse SPY option ticker information from OCC-format filename.

    Recognizes patterns like 'SPY250119C00450000' where:
    - SPY = underlying ticker
    - 250119 = YYMMDD expiry (e.g., Jan 19, 2025)
    - C/P = call/put flag
    - 00450000 = strike × 1000 (e.g., 4500 → $450 strike)

    Returns None for non-matching filenames. Designed for SPY options but pattern
    works for any OCC-format equity option files.

    Examples
    --------
    >>> parse_ticker_info(Path("SPY250119C00450000.csv"))
    {'expiry': '250119', 'expiry_dt': Timestamp('2025-01-19'),
     'cp': 'C', 'strike': 450.0}
    """
    m = _ticker_re.match(path.stem)
    if not m:
        return None
    expiry_str, cp, strike_digits = m.groups()
    strike = int(strike_digits) / 1000.0
    expiry_dt = pd.to_datetime("20" + expiry_str[:2] + expiry_str[2:6], format="%Y%m%d")
    return {"expiry": expiry_str, "expiry_dt": expiry_dt, "cp": cp, "strike": strike}


def compute_ivs_vectorized(
        prices: NDArray[np.float64],
        spots: NDArray[np.float64],
        strikes: NDArray[np.float64],
        ttes: NDArray[np.float64],
        r: float,
        q: float = 0.0,
        flags: NDArray[np.str_] = None
) -> NDArray[np.float64]:
    """Compute Black-Scholes-Merton implied vols from option prices.

    Parameters
    ----------
    prices : NDArray[np.float64]
        Option mid-prices (or bid/ask).
    spots : NDArray[np.float64]
        Underlying spot prices (same length as prices).
    strikes : NDArray[np.float64]
        Strike prices.
    ttes : NDArray[np.float64]
        Time-to-expiry in years.
    r : float
        Risk-free rate (continuous).
    q : float, default 0.0
        Dividend yield (continuous).
    flags : NDArray[np.str_], optional
        'c'/'C' for calls, 'p'/'P' for puts.

    Returns
    -------
    NDArray[np.float64]
        Implied vols (NaN for failures).

    Notes
    -----
    Uses py_vollib BSM solver.
    """
    n = len(prices)
    if flags is None:
        flags = np.full(n, 'c', dtype=np.str_)

    ivs = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not all(np.isfinite(x) and x > 0 for x in [prices[i], spots[i], strikes[i], ttes[i]]):
            continue
        try:
            # Cast to match dtype
            ivs[i] = float(  # ← Explicit float → np.float64
                bsm_iv(
                    float(prices[i]), float(spots[i]), float(strikes[i]),
                    float(ttes[i]), r, q, str(flags[i]).lower()
                )
            )
        except (BelowIntrinsicException, Exception):
            ivs[i] = np.nan
    return ivs


def calculate_implied_forward(
        spot: pd.Series,
        tte: pd.Series,
        r: float,
        strike: pd.Series,
        call_mid: pd.Series,
        put_mid: pd.Series
) -> pd.Series:
    """Compute forward price F from put-call parity on same strike slice.

    PCP: C - P = e^(-rT) * (F - K)
    → F = K + e^(rT) * (C - P)

    Parameters
    ----------
    spot : pd.Series
        Underlying spot price time series.
    tte : pd.Series
        Time-to-expiry (years) for this expiry.
    r : float
        Risk-free rate (constant, continuous).
    strike : pd.Series
        Fixed strike (same value across series).
    call_mid : pd.Series
        Call option mid-prices (same strike/expiry).
    put_mid : pd.Series
        Put option mid-prices (matching call).

    Returns
    -------
    pd.Series
        Implied forwards (NaN where data invalid/missing).

    Notes
    -----
    Assumes same index/length across series. Handles NaNs gracefully.

    Examples
    --------
    >>> import pandas as pd
    >>> spot = pd.Series([100.0, 101.0])
    >>> call_mid = pd.Series([3.2, 3.5])
    >>> put_mid = pd.Series([2.8, 3.0])
    >>> strike = pd.Series([100.0, 100.0])
    >>> tte = pd.Series([0.25, 0.25])
    >>> calculate_implied_forward(spot, tte, 0.05, strike, call_mid, put_mid)
    0    100.50
    1    101.26
    dtype: float64
    """
    fwd = strike + np.exp(r * tte.astype(float)) * (call_mid.astype(float) - put_mid.astype(float))
    mask = (spot > 0) & (tte > 0) & (strike > 0) & call_mid.notna() & put_mid.notna()
    return fwd.where(mask, np.nan)


def choose_leg(strike: float, forward: float, call_mid: float, put_mid: float) -> float:
    """Select most liquid OTM option leg for robust IV surface construction.

    Intuition: OTM options have higher liquidity/depth vs ITM counterparts
    due to hedging demand asymmetry and lower capital requirements. ITM options
    embed intrinsic value → noisier extrinsic/vol pricing.

    Logic:
    * K ≥ F → call OTM (put ITM) → prefer call
    * K < F → put OTM (call ITM) → prefer put
    * Fallback to other leg if NaN/missing

    Improves SVI calibration stability by prioritizing cleaner quotes.

    Parameters
    ----------
    strike : float
        Option strike K.
    forward : float
        Implied forward F_{t,T}.
    call_mid : float
        Call option mid-price.
    put_mid : float
        Put option mid-price.

    Returns
    -------
    float
        Preferred leg price (mid), or other if unavailable.

    Examples
    --------
    >>> choose_leg(105, 100, 2.5, np.nan)  # Call OTM → use call
    2.5
    >>> choose_leg(95, 100, np.nan, 1.8)   # Put OTM → use put
    1.8
    """
    if strike >= forward:
        return call_mid if np.isfinite(call_mid) else put_mid
    return put_mid if np.isfinite(put_mid) else call_mid


def prepare_slice(
        df_slice: pd.DataFrame,
        maturity_col: str = "maturity",
        strike_col: str = "strike",
        iv_col: str = "iv",
        forward_col: str = "implied_forward",
        min_points: int = 5
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], Optional[float]]:
    """Transform single maturity slice to SVI-ready inputs: k, w_target, F.

    Pipeline:
    1. Extract T, F (uniform across slice)
    2. Filter valid (finite, positive) K, σ_mkt
    3. Compute k = log(K/F), w = σ_mkt² × T
    4. Final finite check + extreme k clipping (±10 ~ 0.00005% moneyness)

    Rejects illiquid/noisy slices (< min_points after cleaning).

    Parameters
    ----------
    df_slice : pd.DataFrame
        Single maturity cross-section (strikes × time).
    maturity_col : str, default "maturity"
        Column with T (years fraction).
    strike_col : str, default "strike"
        Strike prices K.
    iv_col : str, default "iv"
        BSM implied vols σ_mkt.
    forward_col : str, default "implied_forward"
        F_{t,T} (constant per slice).
    min_points : int, default 5
        Minimum valid strikes required.

    Returns
    -------
    tuple[NDArray|None, NDArray|None, float|None]
        (k, w_target, F) or (None, None, None) if invalid.

    Notes
    -----
    k clipping prevents optimizer divergence from wing noise.

    Examples
    --------
    >>> df = pd.DataFrame({"strike": [90,100,110], "iv": [0.22,0.20,0.23],
    ...                    "maturity": 0.25, "implied_forward": 100})
    >>> prepare_slice(df)
    (array([-0.105,  0.   ,  0.095]), array([0.012, 0.010, 0.013]), 100.0)
    """
    if df_slice.empty:
        return None, None, None

    T = float(df_slice[maturity_col].iloc[0])
    if T <= 0:
        return None, None, None

    F = float(df_slice[forward_col].iloc[0])
    if not np.isfinite(F) or F <= 0:
        return None, None, None

    K = df_slice[strike_col].to_numpy(dtype=float)
    sigma_mkt = df_slice[iv_col].to_numpy(dtype=float)

    valid = np.isfinite(K) & np.isfinite(sigma_mkt) & (K > 0) & (sigma_mkt > 0)
    if np.sum(valid) < min_points:
        return None, None, None

    K, sigma_mkt = K[valid], sigma_mkt[valid]
    k = np.log(K / F)
    w_target = sigma_mkt ** 2 * T
    finite = np.isfinite(k) & np.isfinite(w_target)
    if np.sum(finite) < min_points:
        return None, None, None

    k = np.clip(k[finite], -10.0, 10.0)
    return k, w_target[finite], F


def calibrate_slice(
        df_slice: pd.DataFrame,
        model: Parametrization,
        maturity_col: str = "maturity",
        **model_kwargs
) -> Optional[Dict[str, float]]:
    """Calibrate parametrization to single maturity cross-section.

    Orchestrates prepare_slice() → model.calibrate() → store F.

    Parameters
    ----------
    df_slice : pd.DataFrame
        Strike slice for one T (from groupby 'maturity').
    model : Parametrization
        SVI/SSVI/eSSVI instance.
    maturity_col : str, default "maturity"
        Passed to prepare_slice().
    **model_kwargs
        Forwarded to model.calibrate().

    Returns
    -------
    Dict[str, float] or None
        Calibrated params + 'forward', or None if prep/calibration fails.

    Examples
    --------
    >>> svi = SVI()
    >>> params = calibrate_slice(df_slice, svi)
    >>> params["a"], params["forward"]
    (0.01, 100.0)
    """
    k, w_target, F = prepare_slice(df_slice)
    if k is None:
        logger.warning("Insufficient data for calibration")
        return None

    params = model.calibrate(k, w_target, **model_kwargs)
    if params is None:
        logger.warning("Calibration failed")
    else:
        params["forward"] = F
    return params


def apply_slice(
        df_slice: pd.DataFrame,
        params: Dict[str, float],
        model: Parametrization,
        maturity_col: str = "maturity",
        strike_col: str = "strike",
        iv_col: str = "iv",
        fitted_col: str = "fitted_iv",
        residual_col: str = "residual_iv"
) -> pd.DataFrame:
    """Generate fitted IVs + residuals for calibrated slice.

    Forward pass: w(k; params) → σ_fit = sqrt(w/T)

    Adds columns in-place to copy. Clamps w/T ≥ 0 for numerical stability.
    Residuals optional (requires original iv_col).

    Parameters
    ----------
    df_slice : pd.DataFrame
        Original slice (strikes for this T).
    params : Dict[str, float]
        From calibrate_slice(), must contain 'forward'.
    model : Parametrization
        Matching model instance.
    maturity_col : str, default "maturity"
        Uniform T value.
    strike_col : str, default "strike"
        K values.
    iv_col : str, default "iv"
        For residuals (ignored if missing).
    fitted_col : str, default "fitted_iv"
        Output fitted σ column name.
    residual_col : str, default "residual_iv"
        Output residual column name.

    Returns
    -------
    pd.DataFrame
        Enriched slice with fitted_iv, residual_iv.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df_slice = pd.DataFrame({
    ...     "strike": np.array([90, 100, 110]),
    ...     "iv": np.array([0.22, 0.20, 0.23]),
    ...     "maturity": 0.25
    ... })
    >>> params = {"forward": 100.0, "a": 0.01, "b": 0.1, "rho": -0.5, "m": 0, "sigma": 0.3}
    >>> svi = SVI()
    >>> fitted_df = apply_slice(df_slice, params, svi)
    >>> fitted_df[["strike", "iv", "fitted_iv", "residual_iv"]].round(3)
       strike    iv  fitted_iv  residual_iv
    0     90.0  0.220     0.219       0.001
    1    100.0  0.200     0.200       0.000
    2    110.0  0.230     0.229       0.001
    """
    T = float(df_slice[maturity_col].iloc[0])
    F = params["forward"]
    K = df_slice[strike_col].to_numpy(dtype=float)
    k = np.log(K / F)

    w_fit = model.total_variance(k, params)
    sigma_fit = np.sqrt(np.maximum(w_fit / T, 0.0))

    out = df_slice.copy()
    out[fitted_col] = sigma_fit
    if iv_col in out:
        out[residual_col] = out[iv_col] - sigma_fit
    return out


# Factory
def get_model(model_name: str) -> Parametrization:
    """Factory for parametrization by lowercase name.

    Supported:
    * 'svi': Raw SVI (Gatheral 2004) - 5 params
    * 'ssvi': Surface SSVI - arbitrage-free across T
    * 'essvi': eSSVI - extended rho(T) parametrization

    Case-insensitive. Extensible: add to dict.

    Parameters
    ----------
    model_name : str
        'svi', 'ssvi', 'essvi'.

    Returns
    -------
    Parametrization
        Instantiated model.

    Raises
    ------
    KeyError
        Unknown model_name.

    Examples
    --------
    >>> svi = get_model("SVI")
    >>> ssvi = get_model("ssvi")
    """
    return {
        "svi": SVI(),
        "ssvi": SSVI(),
        "essvi": ESSVI()
    }[model_name.lower()]
