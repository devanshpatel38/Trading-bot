from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, macd, last_timestamp


class MacdMomentumStrategy(Strategy):
    name = "macd_momentum"

    @staticmethod
    def default_params() -> dict:
        return {"fast": 12, "slow": 26, "signal": 9}

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["slow"] + p["signal"] + 1:
            return self.neutral(df, "insufficient data")
        macd_line, signal_line, hist = macd(df["close"], p["fast"], p["slow"], p["signal"])
        m, s = float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
        h, hp = float(hist.iloc[-1]), float(hist.iloc[-2])
        m_prev, s_prev = float(macd_line.iloc[-2]), float(signal_line.iloc[-2])
        crossed_up = m_prev <= s_prev and m > s
        crossed_dn = m_prev >= s_prev and m < s
        buy = sell = 0.0
        if crossed_up:
            buy, reason = 75.0, "MACD bullish cross"
        elif crossed_dn:
            sell, reason = 75.0, "MACD bearish cross"
        elif m > s:
            buy, reason = 55.0, "MACD above signal, momentum rising"
        elif m < s and h <= hp:
            sell, reason = 55.0, "MACD below signal, momentum falling"
        else:
            reason = "MACD momentum mixed"
        regime = "trending" if max(buy, sell) >= 70.0 else "ranging"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))