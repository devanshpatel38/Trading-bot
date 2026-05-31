import pandas as pd

from hyperbot.strategies.rsi_meanrev import RsiMeanRevStrategy


def _df(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="15min")
    close = pd.Series(values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_falling_price_is_oversold_buy():
    sig = RsiMeanRevStrategy().analyze(_df(list(range(60, 1, -1))))
    assert sig.buy_confidence == 100.0
    assert sig.sell_confidence == 0.0


def test_rising_price_is_overbought_sell():
    sig = RsiMeanRevStrategy().analyze(_df(list(range(1, 60))))
    assert sig.sell_confidence == 100.0
    assert sig.buy_confidence == 0.0
