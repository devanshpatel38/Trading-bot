from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StrategySignal:
    strategy: str
    buy_confidence: float
    sell_confidence: float
    regime: str
    reason: str
    timestamp: object = None

    def __post_init__(self):
        self.buy_confidence = max(0.0, min(100.0, float(self.buy_confidence)))
        self.sell_confidence = max(0.0, min(100.0, float(self.sell_confidence)))


def last_timestamp(df: pd.DataFrame):
    return df.index[-1] if len(df.index) else None


class Strategy(ABC):
    name: str = "base"

    def __init__(self, params: dict | None = None):
        self.params = {**self.default_params(), **(params or {})}

    @staticmethod
    def default_params() -> dict:
        return {}

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        ...

    def neutral(self, df: pd.DataFrame, reason: str) -> StrategySignal:
        return StrategySignal(self.name, 0.0, 0.0, "unknown", reason, last_timestamp(df))