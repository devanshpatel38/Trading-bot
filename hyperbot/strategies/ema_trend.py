from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, ema, atr, last_timestamp


class EmaTrendStrategy(Strategy):
    name = "ema_trend"

    @staticmethod
    def default_params() -> dict:
        return {"fast": 21, "slow": 55, "atr_period": 14, "pullback_atr": 1.0, "trend_min_pct": 0.2}

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["slow"] + 1:
            return self.neutral(df, "insufficient data")
        close = df["close"]
        fast = ema(close, p["fast"])
        slow = ema(close, p["slow"])
        a = atr(df, p["atr_period"])
        c, f, s, av = float(close.iloc[-1]), float(fast.iloc[-1]), float(slow.iloc[-1]), float(a.iloc[-1])
        if pd.isna(av) or av <= 0:
            return self.neutral(df, "ATR unavailable")
        trend_pct = abs(f - s) / c * 100.0
        regime = "trending" if trend_pct >= p["trend_min_pct"] else "ranging"
        dist = (c - f) / av  # signed distance from fast EMA in ATR units
        buy = sell = 0.0
        if f > s:
            if dist <= p["pullback_atr"]:
                proximity = max(0.0, 1.0 - abs(dist) / p["pullback_atr"])
                buy = min(100.0, 40.0 + 60.0 * proximity)
                reason = f"Uptrend (fast>slow, {trend_pct:.2f}%), pullback {dist:.2f} ATR to fast EMA"
            else:
                buy = 20.0
                reason = f"Uptrend but extended {dist:.2f} ATR above fast EMA"
        elif f < s:
            if dist >= -p["pullback_atr"]:
                proximity = max(0.0, 1.0 - abs(dist) / p["pullback_atr"])
                sell = min(100.0, 40.0 + 60.0 * proximity)
                reason = f"Downtrend (fast<slow, {trend_pct:.2f}%), pullback {dist:.2f} ATR to fast EMA"
            else:
                sell = 20.0
                reason = f"Downtrend but extended {dist:.2f} ATR below fast EMA"
        else:
            reason = "No clear trend"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))