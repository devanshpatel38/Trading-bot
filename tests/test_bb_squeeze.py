import pandas as pd

from hyperbot.strategies.bb_squeeze import BbSqueezeStrategy


def _df(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="15min")
    close = pd.Series(values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_flat_series_is_squeeze_upward_bias():
    sig = BbSqueezeStrategy().analyze(_df([100.0] * 80))
    assert sig.regime == "squeeze"
    assert sig.buy_confidence == 50.0
    assert sig.sell_confidence == 0.0


def test_insufficient_data_is_neutral():
    sig = BbSqueezeStrategy().analyze(_df([100.0] * 10))
    assert sig.reason == "insufficient data"
