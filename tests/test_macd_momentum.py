import numpy as np
import pandas as pd

from hyperbot.strategies.macd_momentum import MacdMomentumStrategy

CONF = {0.0, 25.0, 50.0, 75.0, 100.0}


def _df(close_values):
    idx = pd.date_range("2021-01-01", periods=len(close_values), freq="15min")
    close = pd.Series(close_values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def _uptrend_with_fresh_cross():
    """>=201 bars: sustained uptrend (close well above EMA200), a sharp dip that drives
    MACD below its signal, then a sharp rally producing a FRESH bullish cross within
    cross_lookback, with rising histogram and macd_line > 0.

    All four buy components true.
    """
    n = 230
    base = np.linspace(100.0, 260.0, n)
    close = base.copy()
    dip_start = n - 18
    dip_bottom = n - 5
    for i in range(dip_start, dip_bottom + 1):
        frac = (i - dip_start) / (dip_bottom - dip_start)
        close[i] = base[dip_start] - 45.0 * frac
    for i in range(dip_bottom + 1, n):
        close[i] = close[i - 1] + 25.0
    return _df(close)


def test_uptrend_fresh_cross_gives_buy():
    sig = MacdMomentumStrategy().analyze(_uptrend_with_fresh_cross())
    assert sig.buy_confidence >= 75.0
    assert sig.buy_confidence > sig.sell_confidence
    assert sig.regime == "trending"
    assert sig.buy_confidence in CONF
    assert sig.sell_confidence in CONF


def test_insufficient_data_is_neutral():
    sig = MacdMomentumStrategy().analyze(_df(list(np.linspace(100, 200, 50))))
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"
