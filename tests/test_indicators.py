import pandas as pd
import pytest

from hyperbot.strategies.base import ema, rsi, atr, macd, bollinger_bands


def test_ema_matches_hand_computation():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = ema(s, 3)  # alpha = 0.5, adjust=False
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(1.5)
    assert result.iloc[2] == pytest.approx(2.25)
    assert result.iloc[3] == pytest.approx(3.125)
    assert result.iloc[4] == pytest.approx(4.0625)


def test_rsi_extremes_and_value():
    up = pd.Series(range(1, 30), dtype=float)
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(range(30, 1, -1), dtype=float)
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0)
    mixed = pd.Series([1, 2, 4, 3], dtype=float)
    assert rsi(mixed, 2).iloc[-1] == pytest.approx(66.6667, abs=1e-3)


def test_atr_sma_based():
    df = pd.DataFrame({
        "high": [10, 11, 12],
        "low": [8, 9, 10],
        "close": [9, 10, 11],
    }, dtype=float)
    assert atr(df, 2).iloc[-1] == pytest.approx(2.0)


def test_macd_constant_series_is_zero():
    s = pd.Series([100.0] * 60)
    macd_line, signal_line, hist = macd(s)
    assert macd_line.iloc[-1] == pytest.approx(0.0)
    assert signal_line.iloc[-1] == pytest.approx(0.0)
    assert hist.iloc[-1] == pytest.approx(0.0)


def test_bollinger_bands_known_window():
    s = pd.Series([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
    upper, mid, lower = bollinger_bands(s, period=8, num_std=2)
    assert mid.iloc[-1] == pytest.approx(5.0)
    assert upper.iloc[-1] == pytest.approx(9.0)
    assert lower.iloc[-1] == pytest.approx(1.0)
