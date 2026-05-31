import numpy as np
import pandas as pd

from hyperbot.strategies.rsi_meanrev import RsiMeanRevStrategy


def _df(close_values, open_values=None, hl=2.0):
    idx = pd.date_range("2021-01-01", periods=len(close_values), freq="15min")
    close = pd.Series(close_values, dtype=float, index=idx)
    if open_values is None:
        open_ = close
    else:
        open_ = pd.Series(open_values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": open_, "high": close + hl, "low": close - hl, "close": close, "volume": 100.0},
        index=idx,
    )


def _oversold_turning_up():
    """70-bar series ending oversold (RSI<30) with RSI ticking up, near EMA50, bullish candle.

    Long flat region keeps EMA50 near the price level; a modest final dip drives RSI
    oversold while keeping close within 1.5*ATR of EMA50; the last bar ticks up (bullish,
    RSI > RSI[-2]). All four buy components are true.
    """
    n = 70
    close = [196.0] * (n - 7)
    close += [195.0, 194.0, 193.0, 192.2, 191.6, 191.2]
    close.append(191.6)  # last bar ticks up
    open_ = list(close)
    open_[-1] = close[-1] - 0.3  # bullish last candle
    return _df(close, open_, hl=2.0)


def test_oversold_turning_up_gives_buy():
    sig = RsiMeanRevStrategy().analyze(_oversold_turning_up())
    assert sig.buy_confidence >= 75.0
    assert sig.buy_confidence > sig.sell_confidence
    assert sig.regime == "ranging"
    assert sig.buy_confidence in {0.0, 25.0, 50.0, 75.0, 100.0}
    assert sig.sell_confidence in {0.0, 25.0, 50.0, 75.0, 100.0}


def test_insufficient_data_is_neutral():
    sig = RsiMeanRevStrategy().analyze(_df(list(range(60, 20, -1))))  # 40 bars < 51
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"
