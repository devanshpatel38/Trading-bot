import numpy as np
import pandas as pd

from hyperbot.strategies.macd_momentum import MacdMomentumStrategy


def _df(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="15min")
    close = pd.Series(values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_rising_series_gives_buy():
    sig = MacdMomentumStrategy().analyze(_df(list(np.linspace(100, 200, 80))))
    assert sig.buy_confidence > 0.0
    assert sig.sell_confidence == 0.0


def test_insufficient_data_is_neutral():
    sig = MacdMomentumStrategy().analyze(_df([100.0] * 5))
    assert sig.reason == "insufficient data"