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
    """Parse SPY option ticker from filename."""
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
    """Vectorized BSM IV computation with error handling."""
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
    """Implied forward from put-call parity."""
    fwd = strike + np.exp(r * tte.astype(float)) * (call_mid.astype(float) - put_mid.astype(float))
    mask = (spot > 0) & (tte > 0) & (strike > 0) & call_mid.notna() & put_mid.notna()
    return fwd.where(mask, np.nan)


def choose_leg(strike: float, forward: float, call_mid: float, put_mid: float) -> float:
    """Choose OTM leg: call if ITM put, put if ITM call."""
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
    """Prepare k (log-moneyness), w_target (total var), forward for calibration."""
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
    """Calibrate single maturity slice."""
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
    """Apply calibrated params to slice."""
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
    """Get parametrization by name."""
    return {
        "svi": SVI(),
        "ssvi": SSVI(),
        "essvi": ESSVI()
    }[model_name.lower()]
