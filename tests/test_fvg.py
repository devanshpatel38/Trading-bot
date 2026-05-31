import numpy as np
import pandas as pd

from hyperbot.strategies.fvg import FvgStrategy

CONF = {0.0, 25.0, 50.0, 75.0, 100.0}


def _unfilled_bullish_fvg():
    """Construct a recent UNFILLED bullish FVG within the lookback, current price within
    0.25 ATR of the gap, bullish last candle.

    Bullish gap at j: low[j] > high[j-2]; zone = (high[j-2], low[j]) = (101, 103).
    All four buy components true: gap exists, unfilled, proximity, bullish confirmation.
    """
    n = 30
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)

    j = n - 4
    # displacement candle (j-1) and gap candle (j): low[j]=103 > high[j-2]=101
    high[j - 1], low[j - 1], close[j - 1], open_[j - 1] = 104.0, 100.5, 103.5, 100.5
    high[j], low[j], close[j], open_[j] = 105.0, 103.0, 104.0, 103.5
    # bars after j stay above zone_top (103) so the gap remains UNFILLED
    for k in range(j + 1, n):
        high[k], low[k], close[k], open_[k] = 104.0, 103.2, 103.5, 103.4
    # final bar near the zone (within 0.25 ATR of top edge 103) and bullish
    high[-1], low[-1], close[-1], open_[-1] = 103.5, 103.1, 103.2, 103.0

    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0}, index=idx
    )


def test_unfilled_bullish_fvg_gives_buy():
    sig = FvgStrategy().analyze(_unfilled_bullish_fvg())
    assert sig.buy_confidence >= 75.0
    assert sig.sell_confidence == 0.0
    assert sig.regime == "imbalance"
    assert sig.buy_confidence in CONF


def test_no_gap_is_balanced():
    n = 30
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    close = pd.Series([100.0] * n, index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )
    sig = FvgStrategy().analyze(df)
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.regime == "balanced"


def test_insufficient_data_is_neutral():
    n = 5
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    close = pd.Series([100.0] * n, index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )
    sig = FvgStrategy().analyze(df)
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"