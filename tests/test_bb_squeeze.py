import numpy as np
import pandas as pd

from hyperbot.strategies.bb_squeeze import BbSqueezeStrategy

CONF = {0.0, 25.0, 50.0, 75.0, 100.0}


def _df(close_values, volume_values=None):
    idx = pd.date_range("2021-01-01", periods=len(close_values), freq="15min")
    close = pd.Series(close_values, dtype=float, index=idx)
    if volume_values is None:
        vol = pd.Series(100.0, index=idx)
    else:
        vol = pd.Series(volume_values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": vol},
        index=idx,
    )


def _squeeze_then_breakout():
    """>=70 bars: long perfectly flat (zero-volatility) squeeze, then a final bar that
    closes above the upper band on above-average volume with rising bandwidth.

    All four buy components true:
      squeeze (bw[-2] <= 25th pct), expansion (bw[-1] > bw[-2]),
      breakout (close > upper), vol_ok (volume[-1] > vol_ma[-1]).
    """
    n = 75
    close = np.full(n, 100.0)
    vol = np.full(n, 100.0)
    close[-1] = 105.0  # breakout above the upper band
    vol[-1] = 300.0    # above the 20-bar volume average
    return _df(close, vol)


def test_squeeze_breakout_gives_buy():
    sig = BbSqueezeStrategy().analyze(_squeeze_then_breakout())
    assert sig.buy_confidence >= 75.0
    assert sig.buy_confidence > sig.sell_confidence
    assert sig.regime == "expansion"
    assert sig.buy_confidence in CONF
    assert sig.sell_confidence in CONF


def test_insufficient_data_is_neutral():
    sig = BbSqueezeStrategy().analyze(_df([100.0] * 10))
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"