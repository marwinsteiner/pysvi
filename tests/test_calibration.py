from src.pysvi.calibration import *


def test_parse_ticker_info():
    from pathlib import Path
    result = parse_ticker_info(Path("SPY250119C00450000.csv"))
    expected = {
        "expiry": "250119", "cp": "C", "strike": 450.0,
        "expiry_dt": pd.Timestamp("2025-01-19 00:00:00")
    }
    assert result == expected
    assert parse_ticker_info(Path("invalid.txt")) is None


def test_choose_leg_otm_preference():
    """Prefers OTM liquidity, falls back gracefully."""
    assert choose_leg(105, 100, 2.5, np.nan) == 2.5  # Call OTM ✓
    assert choose_leg(95, 100, np.nan, 1.8) == 1.8  # Put OTM ✓
    assert choose_leg(105, 100, np.nan, 3.1) == 3.1  # Fallback ✓
    assert np.isnan(choose_leg(100, 100, np.nan, np.nan))  # None available


def test_prepare_slice_valid(atm_slice):
    k, w_target, F = prepare_slice(atm_slice)
    assert k.shape == (21,)
    assert np.all(np.isfinite(k))
    assert np.all((k > -10) & (k < 10))
    assert np.all(w_target > 0)
    assert F == 100.0


def test_prepare_slice_rejects_empty():
    empty_df = pd.DataFrame(columns=["strike", "iv", "maturity", "implied_forward"])
    result = prepare_slice(empty_df)
    assert result == (None, None, None)


def test_prepare_slice_rejects_insufficient(valid_slice_for_prepare, invalid_slice_for_prepare):
    # Valid passes
    k, _, _ = prepare_slice(valid_slice_for_prepare)
    assert k is not None and len(k) >= 5

    # Invalid rejects
    result = prepare_slice(invalid_slice_for_prepare, min_points=3)
    assert result == (None, None, None)


def test_calculate_implied_forward():
    spot = pd.Series([100.0, 101.0])
    tte = pd.Series([0.25, 0.25])
    strike = pd.Series([100.0, 100.0])
    call_mid = pd.Series([3.2, 3.5])
    put_mid = pd.Series([2.8, 3.0])
    fwd = calculate_implied_forward(spot, tte, 0.05, strike, call_mid, put_mid)
    expected = pd.Series([100.41, 100.51])
    pd.testing.assert_series_equal(fwd.round(2), expected.round(2))


def test_full_svi_roundtrip(calibrated_svi):
    """End-to-end SVI: calibrate → apply → RMSE < 50bps."""
    model, df_slice, params = calibrated_svi
    fitted = apply_slice(df_slice, params, model)
    residuals = fitted["residual_iv"].dropna()
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    assert 0 <= rmse < 0.005, f"RMSE {rmse:.4f} > 0.005"


def test_ssvi_roundtrip(ssvi_calibrated):
    """End-to-end SSVI roundtrip."""
    model, df_slice, params = ssvi_calibrated
    fitted = apply_slice(df_slice, params, model)
    rmse = float(np.sqrt(np.mean(fitted["residual_iv"] ** 2)))
    assert rmse < 0.025


def test_essvi_roundtrip(essvi_calibrated):
    """End-to-end eSSVI roundtrip."""
    model, df_slice, params = essvi_calibrated
    fitted = apply_slice(df_slice, params, model)
    rmse = float(np.sqrt(np.mean(fitted["residual_iv"] ** 2)))
    assert rmse < 0.025

