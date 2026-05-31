import pandas as pd

from hyperbot.strategies.fvg import FvgStrategy


def test_bullish_gap_gives_buy():
    n = 20
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    high = [101.0] * n
    low = [99.0] * n
    close = [100.0] * n
    # Create a bullish gap: last candle's low (110) above candle[-3] high (101)
    high[-1], low[-1], close[-1] = 112.0, 110.0, 111.0
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 100.0}, index=idx)
    sig = FvgStrategy().analyze(df)
    assert sig.buy_confidence > 0.0
    assert sig.sell_confidence == 0.0
    assert sig.regime == "imbalance"


def test_no_gap_is_balanced():
    n = 20
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    close = pd.Series([100.0] * n, index=idx)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0}, index=idx)
    sig = FvgStrategy().analyze(df)
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.regime == "balanced"