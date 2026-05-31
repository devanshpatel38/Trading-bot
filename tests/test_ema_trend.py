import numpy as np
import pandas as pd

from hyperbot.strategies.ema_trend import EmaTrendStrategy


def _ramp_df(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="15min")
    close = pd.Series(values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_uptrend_pullback_gives_buy():
    values = list(np.linspace(100, 200, 120))
    values[-1] = values[-2] - 1.0  # small pullback on last bar
    sig = EmaTrendStrategy().analyze(_ramp_df(values))
    assert sig.buy_confidence > sig.sell_confidence
    assert sig.regime == "trending"


def test_insufficient_data_is_neutral():
    sig = EmaTrendStrategy().analyze(_ramp_df([100, 101, 102]))
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"
