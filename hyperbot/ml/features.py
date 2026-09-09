"""Causal market-context features, aligned to the candle index.

Every column is backward-looking (uses only bars <= t), so a feature read at a
trade's entry bar carries no lookahead. These are the "Market context" family:
trend state + volatility state at entry. The "Signal internals" family (agreement
count, per-strategy confidences, side) is read straight off the trade record in
dataset.py — it needs no recomputation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategies.base import atr, bollinger_bands, ema, rsi


def market_features(
    df: pd.DataFrame,
    htf_period: int = 800,
    atr_period: int = 14,
    slope_lookback: int = 24,
    atr_pctile_window: int = 200,
    rv_window: int = 24,
    bb_period: int = 20,
) -> pd.DataFrame:
    """Return a DataFrame of market-context features on df.index.

    All indicators are the project's own SMA/EMA-based causal indicators, so a row
    at time t depends only on candles up to and including t.
    """
    close = df["close"]
    a = atr(df, atr_period)
    e = ema(close, htf_period)
    upper, mid, lower = bollinger_bands(close, bb_period)
    logret = np.log(close / close.shift(1))

    feats = pd.DataFrame(index=df.index)
    # how far above/below the EMA800 trend, in ATR units (stretched vs coiled)
    feats["dist_ema800_atr"] = (close - e) / a
    # trend slope over the last `slope_lookback` bars, in ATR units (rising/falling)
    feats["ema800_slope_atr"] = (e - e.shift(slope_lookback)) / a
    # volatility level and its recent percentile rank (calm vs violent)
    feats["atr_pct"] = a / close
    feats["atr_percentile"] = a.rolling(atr_pctile_window).apply(
        lambda x: float((x <= x[-1]).mean()), raw=True
    )
    feats["realized_vol_24"] = logret.rolling(rv_window).std()
    # Bollinger width (squeeze vs expansion) and momentum context
    feats["bb_width"] = (upper - lower) / mid
    feats["rsi14"] = rsi(close, 14)
    feats["ret_24"] = close / close.shift(rv_window) - 1.0
    return feats


MARKET_COLS = [
    "dist_ema800_atr",
    "ema800_slope_atr",
    "atr_pct",
    "atr_percentile",
    "realized_vol_24",
    "bb_width",
    "rsi14",
    "ret_24",
]
