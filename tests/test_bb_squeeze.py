import numpy as np
import pandas as pd

from hyperbot.strategies.bb_squeeze import BbSqueezeStrategy


def _df(open_v, high_v, low_v, close_v, vol_v):
    n = len(close_v)
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": pd.Series(open_v, dtype=float, index=idx),
            "high": pd.Series(high_v, dtype=float, index=idx),
            "low": pd.Series(low_v, dtype=float, index=idx),
            "close": pd.Series(close_v, dtype=float, index=idx),
            "volume": pd.Series(vol_v, dtype=float, index=idx),
        },
        index=idx,
    )


def _squeeze_breakout_base(direction="up", last_vol=250.0):
    """>=70 bars. A moderately volatile region 20-50 bars back lifts the 30th-percentile
    of BB width, followed by a tight flat squeeze, then a modest final bar that breaks
    the band while keeping width[-1] still in the bottom 30%.

    With last_vol >= 2x avg this fires all four directional components:
      squeeze=25, breakout=25, volume=25, candle=25.
    """
    n = 80
    close = np.full(n, 100.0) + np.tile([0.05, -0.05], n // 2)[:n]
    # Elevated-volatility window so the 30th-pct width threshold is high enough that the
    # tight squeeze (and the modest breakout bar) still sit in the bottom 30%.
    for i in range(n - 50, n - 30):
        close[i] = 100.0 + (5.0 if i % 2 == 0 else -5.0)
    close[-30:-1] = 100.0
    close[-2] = 100.0

    open_ = close.copy()
    high = close + 0.1
    low = close - 0.1
    vol = np.full(n, 100.0)

    if direction == "up":
        close[-1] = 101.2
        open_[-1] = 100.0  # bullish candle
        high[-1] = 101.5
        low[-1] = 99.9
    else:
        close[-1] = 98.8
        open_[-1] = 100.0  # bearish candle
        high[-1] = 100.1
        low[-1] = 98.5
    vol[-1] = last_vol
    return _df(open_, high, low, close, vol)


def test_squeeze_breakout_up_gives_full_buy():
    sig = BbSqueezeStrategy().analyze(_squeeze_breakout_base("up"))
    assert sig.buy_confidence == 100.0
    # squeeze(25) + volume(25) are shared; breakout-down and candle-down do not fire.
    assert sig.sell_confidence == 50.0
    assert sig.regime == "expansion"
    assert "squeeze=25" in sig.reason
    assert "breakout=25" in sig.reason
    assert "volume=25" in sig.reason
    assert "candle=25" in sig.reason


def test_squeeze_breakout_down_gives_full_sell():
    sig = BbSqueezeStrategy().analyze(_squeeze_breakout_base("down"))
    assert sig.sell_confidence == 100.0
    assert sig.buy_confidence == 50.0
    assert sig.regime == "expansion"


def test_volume_proportional_continuous():
    """Volume that is not >=2x avg yields a CONTINUOUS (non-multiple-of-25) score.
    last_vol=150 with a 100-baseline -> rolling avg ~102.5 -> ratio ~1.463 ->
    25 * (1.463-1) ~= 11.585 volume points."""
    sig = BbSqueezeStrategy().analyze(_squeeze_breakout_base("up", last_vol=150.0))
    # squeeze 25 + breakout 25 + candle 25 + volume ~11.585
    assert abs(sig.buy_confidence - 86.585) < 0.01
    assert "volume=11.585" in sig.reason


def test_volume_zero_when_below_avg():
    sig = BbSqueezeStrategy().analyze(_squeeze_breakout_base("up", last_vol=80.0))
    assert sig.buy_confidence == 75.0  # squeeze + breakout + candle
    assert "volume=0" in sig.reason


def test_squeeze_no_breakout_regime():
    """Flat squeeze, no breakout on the last bar -> regime 'squeeze', breakout=0."""
    n = 80
    close = np.full(n, 100.0) + np.tile([0.05, -0.05], n // 2)[:n]
    for i in range(n - 50, n - 30):
        close[i] = 100.0 + (5.0 if i % 2 == 0 else -5.0)
    close[-30:] = 100.0
    open_ = close - 0.02  # tiny bullish candle, no breakout
    high = close + 0.1
    low = close - 0.1
    vol = np.full(n, 100.0)
    sig = BbSqueezeStrategy().analyze(_df(open_, high, low, close, vol))
    assert sig.regime == "squeeze"
    assert "breakout=0" in sig.reason


def test_insufficient_data_is_neutral():
    n = 60  # < period(20) + squeeze_lookback(50) = 70
    df = _df([100.0] * n, [100.5] * n, [99.5] * n, [100.0] * n, [100.0] * n)
    sig = BbSqueezeStrategy().analyze(df)
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"
