import numpy as np
import pandas as pd

from hyperbot.strategies.fvg import FvgStrategy

CONF = {0.0, 25.0, 50.0, 75.0, 100.0}


def _unfilled_bullish_fvg():
    """Construct a long UPTREND (>=201 bars) so close > EMA200 (htf true), containing a
    recent UNFILLED bullish FVG within the last 15 bars, current price within 0.25 ATR of
    the gap edge.

    Bullish gap at j: low[j] > high[j-2]; zone = (high[j-2], low[j]).
    All four buy components true: unfilled, proximity, freshness, htf.
    """
    n = 220
    # Steadily rising baseline so the last close sits well above EMA200.
    base = np.linspace(50.0, 250.0, n)
    high = base + 1.0
    low = base - 1.0
    close = base.copy()
    open_ = base.copy()

    j = n - 4  # recent gap, age = (n-1)-j = 3 bars <= freshness_bars (15)
    # gap candle j with low[j] > high[j-2]
    h2 = float(high[j - 2])
    zone_top = h2 + 0.3  # low[j]; small jump so price stays within 0.25 ATR of the edge
    high[j], low[j], close[j], open_[j] = zone_top + 0.2, zone_top, zone_top + 0.1, zone_top + 0.05
    # bars after j hold a flat plateau just above zone_top so the gap stays UNFILLED
    # (close > zone_top) and no NEW gap forms (overlapping ranges, equal highs/lows).
    for k in range(j + 1, n):
        high[k] = zone_top + 0.2
        low[k] = zone_top + 0.05
        close[k] = zone_top + 0.1
        open_[k] = zone_top + 0.08

    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": pd.Series(open_, index=idx),
            "high": pd.Series(high, index=idx),
            "low": pd.Series(low, index=idx),
            "close": pd.Series(close, index=idx),
            "volume": pd.Series(np.full(n, 100.0), index=idx),
        },
        index=idx,
    )


def test_unfilled_bullish_fvg_gives_buy():
    sig = FvgStrategy().analyze(_unfilled_bullish_fvg())
    assert sig.buy_confidence >= 75.0
    assert sig.sell_confidence == 0.0
    assert sig.regime == "imbalance"
    assert sig.buy_confidence in CONF


def test_no_gap_is_balanced():
    n = 210
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    close = pd.Series(np.full(n, 100.0), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": pd.Series(np.full(n, 100.0), index=idx),
        },
        index=idx,
    )
    sig = FvgStrategy().analyze(df)
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.regime == "balanced"


def test_insufficient_data_is_neutral():
    n = 50
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    close = pd.Series(np.full(n, 100.0), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": pd.Series(np.full(n, 100.0), index=idx),
        },
        index=idx,
    )
    sig = FvgStrategy().analyze(df)
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"