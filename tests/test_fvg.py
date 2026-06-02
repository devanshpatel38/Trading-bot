import numpy as np
import pandas as pd
import pytest

from hyperbot.strategies.fvg import FvgStrategy


def _mk(open_v, high_v, low_v, close_v):
    n = len(close_v)
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": pd.Series(open_v, dtype=float, index=idx),
            "high": pd.Series(high_v, dtype=float, index=idx),
            "low": pd.Series(low_v, dtype=float, index=idx),
            "close": pd.Series(close_v, dtype=float, index=idx),
            "volume": pd.Series(np.full(n, 100.0), index=idx),
        },
        index=idx,
    )


def _bullish_fvg(*, age=3, candle_up=True, fill=False):
    """>=201-bar UPTREND (close well above EMA200 -> htf true) with a recent UNFILLED
    bullish FVG. zone = (high[j-2], low[j]); after the gap, price holds a flat plateau
    sitting exactly on the upper edge (low == zone_top) so distance-to-edge is 0 ->
    proximity = full 35.

    age          -> gap age (n-1 - j); >15 makes freshness 0.
    candle_up    -> last bar close>open (candle component 15) or not (0).
    fill         -> drive a later close below zone_top so the gap is FILLED (-> 0/0).
    """
    n = 220
    base = np.linspace(50.0, 250.0, n)
    high = base + 1.0
    low = base - 1.0
    close = base.copy()
    open_ = base.copy()

    j = n - 1 - age
    h2 = float(high[j - 2])
    zone_top = h2 + 0.4  # low[j]; gap = 0.4 (>= 0.08*ATR with ATR~0.3)
    high[j], low[j], close[j], open_[j] = zone_top + 0.5, zone_top, zone_top + 0.2, zone_top + 0.1
    for k in range(j + 1, n):
        high[k] = zone_top + 0.3
        low[k] = zone_top + 0.05
        close[k] = zone_top  # exactly on the upper edge -> dist 0 -> proximity full
        open_[k] = zone_top - 0.02 if candle_up else zone_top + 0.02
    if fill:
        # one close after j re-enters below the gap top -> filled
        close[j + 1] = zone_top - 0.3
    return _mk(open_, high, low, close)


def _bearish_fvg():
    """>=201-bar DOWNTREND (close below EMA200 -> htf true for a short) with a recent
    UNFILLED bearish FVG. zone = (high[j], low[j-2]); price rests on the lower edge."""
    n = 220
    base = np.linspace(250.0, 50.0, n)
    high = base + 1.0
    low = base - 1.0
    close = base.copy()
    open_ = base.copy()

    j = n - 4
    l2 = float(low[j - 2])
    zone_bottom = l2 - 0.4  # high[j]
    high[j], low[j], close[j], open_[j] = zone_bottom, zone_bottom - 0.5, zone_bottom - 0.2, zone_bottom - 0.1
    for k in range(j + 1, n):
        high[k] = zone_bottom
        low[k] = zone_bottom - 0.3
        close[k] = zone_bottom  # on the lower edge -> dist 0 -> proximity full
        open_[k] = zone_bottom + 0.02  # bearish candle (close<open)
    return _mk(open_, high, low, close)


def test_bullish_fvg_all_components_full():
    """Fresh, unfilled, on-edge, uptrend, bullish candle -> 35+25+25+15 = 100 buy."""
    sig = FvgStrategy().analyze(_bullish_fvg())
    assert sig.buy_confidence == 100.0
    assert sig.sell_confidence == 0.0
    assert sig.regime == "imbalance"
    assert "prox=35" in sig.reason
    assert "fresh=25" in sig.reason
    assert "htf=25" in sig.reason
    assert "candle=15" in sig.reason


def test_bearish_fvg_gives_full_sell():
    sig = FvgStrategy().analyze(_bearish_fvg())
    assert sig.sell_confidence == 100.0
    assert sig.buy_confidence == 0.0
    assert sig.regime == "imbalance"
    assert "-> sell 100" in sig.reason


def test_stale_gap_loses_freshness():
    """Gap older than fresh_bars(15) -> freshness 0; the rest still scores."""
    sig = FvgStrategy().analyze(_bullish_fvg(age=18))
    assert "fresh=0" in sig.reason
    # prox 35 + htf 25 + candle 15, no freshness
    assert sig.buy_confidence == 75.0


def test_opposing_candle_zeroes_candle_component():
    """A bearish last candle inside a bullish setup drops the 15-pt candle component."""
    sig = FvgStrategy().analyze(_bullish_fvg(candle_up=False))
    assert "candle=0" in sig.reason
    # prox 35 + fresh 25 + htf 25, no candle
    assert sig.buy_confidence == 85.0


def test_filled_gap_is_balanced():
    sig = FvgStrategy().analyze(_bullish_fvg(fill=True))
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.regime == "balanced"
    assert sig.reason == "FVG filled"


def test_proximity_scales_linearly_with_distance():
    """When price sits part-way out of the gap, proximity is a CONTINUOUS value strictly
    between 0 and 35 (not a multiple of 25). We build a clean, ATR-stable plateau so the
    distance->points ramp is exercised: pts = 35*clamp(1 - dist/(2*proximity_atr*ATR),0,1).
    """
    n = 220
    # Flat baseline far below the eventual close so EMA200 stays under price (htf true).
    base = np.full(n, 100.0)
    high = base + 0.15
    low = base - 0.15
    close = base.copy()
    open_ = base.copy()

    # Push the gap back to bar n-15 so the last bar's 14-period ATR window sits entirely
    # on the post-gap plateau (stable ATR), while the gap age (14) is still <= fresh_bars.
    j = n - 15
    h2 = float(high[j - 2])
    zone_top = h2 + 0.4  # gap 0.4
    # Bridge bar j-1 reaches up to zone_top so the plateau does not open a *second* gap.
    high[j - 1] = zone_top
    high[j], low[j], close[j], open_[j] = zone_top + 0.2, zone_top, zone_top + 0.1, zone_top + 0.05
    # Plateau range 0.2 -> ATR 0.2 -> span = 2*0.25*0.2 = 0.1. close 0.05 above the edge
    # -> dist/span = 0.5 -> proximity = 35*(1-0.5) = 17.5 (a clean continuous value).
    for k in range(j + 1, n):
        high[k] = zone_top + 0.15
        low[k] = zone_top - 0.05
        close[k] = zone_top + 0.05
        open_[k] = zone_top + 0.02  # bullish candle
    sig = FvgStrategy().analyze(_mk(open_, high, low, close))
    # proximity strictly partial
    assert 0.0 < sig.buy_confidence < 100.0
    assert "prox=17.5" in sig.reason
    # 17.5 + fresh 25 + htf 25 + candle 15 = 82.5
    assert sig.buy_confidence == pytest.approx(82.5)


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
    assert sig.reason == "No FVG"


def test_tiny_gap_below_min_gap_atr_ignored():
    """A bullish gap smaller than min_gap_atr*ATR is not counted -> No FVG."""
    n = 220
    base = np.full(n, 100.0)
    high = base + 0.5
    low = base - 0.5  # ATR ~1.0
    close = base.copy()
    open_ = base.copy()
    j = n - 4
    h2 = float(high[j - 2])  # 100.5
    # gap of only 0.02 (< 0.08*ATR=0.08) -> ignored
    zone_top = h2 + 0.02
    high[j], low[j], close[j], open_[j] = zone_top + 0.4, zone_top, zone_top + 0.1, zone_top + 0.05
    for k in range(j + 1, n):
        high[k] = zone_top + 0.4
        low[k] = zone_top - 0.4
        close[k] = zone_top + 0.1
        open_[k] = zone_top + 0.05
    sig = FvgStrategy().analyze(_mk(open_, high, low, close))
    assert sig.reason == "No FVG"
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
