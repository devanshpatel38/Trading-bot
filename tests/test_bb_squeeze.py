import numpy as np
import pandas as pd

from hyperbot.strategies.bb_squeeze import BbSqueezeStrategy

CONF = {0.0, 25.0, 50.0, 75.0, 100.0}


def _df(close_values):
    idx = pd.date_range("2021-01-01", periods=len(close_values), freq="15min")
    close = pd.Series(close_values, dtype=float, index=idx)
    vol = pd.Series(100.0, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": vol},
        index=idx,
    )


def _uptrend_squeeze_breakout():
    """>=210 bars: a long, steady uptrend (so close stays above EMA200 -> htf true),
    a flat low-volatility squeeze near the end, then a final bar that breaks out
    above the upper band with rising bandwidth.

    All four buy components true:
      squeeze (bw[-2] <= 25th pct), expansion (bw[-1] > bw[-2]),
      breakout (close > upper), htf (close > ema200).
    """
    n = 210
    # Gentle linear uptrend so the latest price sits well above the EMA200.
    close = np.linspace(100.0, 300.0, n)
    # Flatten the last stretch into a tight squeeze (constant price) so bandwidth
    # collapses, then break out on the final bar.
    close[-30:] = close[-31]  # perfectly flat squeeze region
    close[-1] = close[-2] + 8.0  # breakout above the upper band, bw rises
    return _df(close)


def test_uptrend_squeeze_breakout_gives_buy():
    sig = BbSqueezeStrategy().analyze(_uptrend_squeeze_breakout())
    # All four buy components fire: squeeze + expansion + breakout-up + htf-up = 100.
    assert sig.buy_confidence == 100.0
    assert sig.buy_confidence > sig.sell_confidence
    assert sig.regime == "expansion"
    # squeeze + expansion are shared (non-directional); the direction-specific
    # breakout-down and htf-down must NOT fire, so sell stays at the 50 floor.
    assert sig.sell_confidence == 50.0
    assert sig.buy_confidence in CONF
    assert sig.sell_confidence in CONF


def test_insufficient_data_is_neutral():
    sig = BbSqueezeStrategy().analyze(_df([100.0] * 50))
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"
