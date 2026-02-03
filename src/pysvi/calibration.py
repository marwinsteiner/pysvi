# src/pysvi/calibration.py
"""
High-level calibration pipeline for IV surfaces from option panels.
Supports SVI, SSVI, eSSVI via models.Parametrization classes.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional
import re

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from py_lets_be_rational.exceptions import BelowIntrinsicException
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as bsm_iv

from .models import SVI, SSVI, ESSVI

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
        prices: np.ndarray,
        spots: np.ndarray,
        strikes: np.ndarray,
        ttes: np.ndarray,
        r: float,
        q: float = 0.0,
        flags: np.ndarray = None
) -> np.ndarray:
    """Vectorized BSM IV computation with error handling."""
    if flags is None:
        flags = np.full(len(prices), 'c')  # Default call

    ivs = np.full(prices.shape, np.nan, dtype=float)
    for i in range(len(prices)):
        if not (np.isfinite(prices[i]) and prices[i] > 0 and
                np.isfinite(spots[i]) and spots[i] > 0 and
                np.isfinite(strikes[i]) and strikes[i] > 0 and
                np.isfinite(ttes[i]) and ttes[i] > 0):
            continue
        try:
            ivs[i] = bsm_iv(
                prices[i], spots[i], strikes[i], ttes[i], r, q, str(flags[i]).lower()
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