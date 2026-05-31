import numpy as np
import pandas as pd

from hyperbot.strategies.base import ema
from hyperbot.strategies.ema_trend import EmaTrendStrategy


def _df(close_values, open_values=None):
    idx = pd.date_range("2021-01-01", periods=len(close_values), freq="15min")
    close = pd.Series(close_values, dtype=float, index=idx)
    if open_values is None:
        open_ = close
    else:
        open_ = pd.Series(open_values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": open_, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def _strong_uptrend():
    """201+ bar strong uptrend where the last bar pulls back near EMA20 and closes bullish.

    Constructed so all four buy components are true:
      close>EMA200, EMA20>EMA200, |close-EMA20|<=0.5*ATR, close>open.
    """
    n = 220
    close = np.linspace(100.0, 320.0, n)
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    # Pull the last close back toward EMA20 (computed on the ramp) so it sits in the pullback zone.
    target = float(ema(pd.Series(close, index=idx), 20).iloc[-1])
    close[-1] = target + 0.2
    open_ = close.copy()
    open_[-1] = close[-1] - 0.3  # bullish last candle (close>open)
    return _df(close, open_)


def test_strong_uptrend_pullback_gives_buy():
    sig = EmaTrendStrategy().analyze(_strong_uptrend())
    assert sig.buy_confidence >= 75.0
    assert sig.buy_confidence > sig.sell_confidence
    assert sig.regime == "trending"
    # confidences are multiples of 25
    assert sig.buy_confidence in {0.0, 25.0, 50.0, 75.0, 100.0}
    assert sig.sell_confidence in {0.0, 25.0, 50.0, 75.0, 100.0}


def test_insufficient_data_is_neutral():
    sig = EmaTrendStrategy().analyze(_df(list(np.linspace(100, 200, 50))))
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"
