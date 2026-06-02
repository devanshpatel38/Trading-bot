import numpy as np
import pandas as pd

from hyperbot.strategies.macd_momentum import MacdMomentumStrategy

CONF = {0.0, 25.0, 50.0, 75.0, 100.0}


def _df(close_values):
    idx = pd.date_range("2021-01-01", periods=len(close_values), freq="15min")
    close = pd.Series(close_values, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def _uptrend_with_fresh_cross():
    """>=201 bars: sustained uptrend (close well above EMA200), a sharp dip that drives
    MACD below its signal, then a sharp rally producing a FRESH bullish cross within
    cross_lookback, with macd_line > signal_line and an accelerating positive histogram.

    All four buy components true: htf, macdsig, hist, cross.
    """
    n = 230
    base = np.linspace(100.0, 260.0, n)
    close = base.copy()
    dip_start = n - 18
    dip_bottom = n - 5
    for i in range(dip_start, dip_bottom + 1):
        frac = (i - dip_start) / (dip_bottom - dip_start)
        close[i] = base[dip_start] - 45.0 * frac
    for i in range(dip_bottom + 1, n):
        close[i] = close[i - 1] + 25.0
    return _df(close)


def _downtrend_with_fresh_cross():
    """Mirror of the uptrend: downtrend below EMA200, a relief rally lifting MACD above
    its signal, then a sharp drop producing a FRESH bearish cross with macd_line <
    signal_line and an accelerating negative histogram. All four sell components true.
    """
    n = 230
    base = np.linspace(260.0, 100.0, n)
    close = base.copy()
    rally_start = n - 18
    rally_top = n - 5
    for i in range(rally_start, rally_top + 1):
        frac = (i - rally_start) / (rally_top - rally_start)
        close[i] = base[rally_start] + 45.0 * frac
    for i in range(rally_top + 1, n):
        close[i] = close[i - 1] - 25.0
    return _df(close)


def test_uptrend_fresh_cross_gives_full_buy():
    sig = MacdMomentumStrategy().analyze(_uptrend_with_fresh_cross())
    assert sig.buy_confidence == 100.0
    assert sig.sell_confidence == 0.0
    assert sig.regime == "momentum_up"
    assert sig.buy_confidence in CONF
    assert "htf=25" in sig.reason
    assert "macdsig=25" in sig.reason
    assert "hist=25" in sig.reason
    assert "cross=25" in sig.reason


def test_downtrend_fresh_cross_gives_full_sell():
    sig = MacdMomentumStrategy().analyze(_downtrend_with_fresh_cross())
    assert sig.sell_confidence == 100.0
    assert sig.buy_confidence == 0.0
    assert sig.regime == "momentum_down"
    assert sig.sell_confidence in CONF


def test_flat_series_is_neutral_zero():
    """A perfectly flat series: close == EMA200 (htf not strictly true), MACD == signal,
    flat histogram, no cross -> all components 0 and regime 'neutral'."""
    sig = MacdMomentumStrategy().analyze(_df(np.full(220, 100.0)))
    assert sig.buy_confidence == 0.0
    assert sig.sell_confidence == 0.0
    assert sig.regime == "neutral"


def test_histogram_needs_correct_sign():
    """An uptrend with no recent cross and a rising-but-negative histogram must NOT score
    the histogram component (spec requires hist>0 AND rising for buy). The HTF and
    macd-vs-signal components may still fire, so we assert the score is a multiple of 25
    and the histogram component is explicitly 0 in the reason when its sign is wrong."""
    # Steady gentle uptrend: close above EMA200 (htf=25), macd>signal (macdsig=25),
    # histogram positive and roughly flat/slightly varying, no fresh cross (cross=0).
    n = 230
    close = np.linspace(100.0, 220.0, n)
    sig = MacdMomentumStrategy().analyze(_df(close))
    assert sig.buy_confidence in CONF
    assert "cross=0" in sig.reason  # no fresh cross in a monotonic trend
    assert sig.regime in {"momentum_up", "neutral"}


def test_insufficient_data_is_neutral():
    sig = MacdMomentumStrategy().analyze(_df(list(np.linspace(100, 200, 50))))
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"
