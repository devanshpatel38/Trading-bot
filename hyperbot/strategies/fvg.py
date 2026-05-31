from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, atr, last_timestamp


class FvgStrategy(Strategy):
    name = "fvg"

    @staticmethod
    def default_params() -> dict:
        return {"atr_period": 14, "min_gap_atr": 0.25}

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < max(4, p["atr_period"] + 1):
            return self.neutral(df, "insufficient data")
        high, low = df["high"], df["low"]
        a = float(atr(df, p["atr_period"]).iloc[-1])
        if pd.isna(a) or a <= 0:
            return self.neutral(df, "ATR unavailable")
        high_2, low_2 = float(high.iloc[-3]), float(low.iloc[-3])
        cur_low, cur_high = float(low.iloc[-1]), float(high.iloc[-1])
        buy = sell = 0.0
        regime, reason = "balanced", "No FVG"
        if cur_low > high_2:
            gap_atr = (cur_low - high_2) / a
            if gap_atr >= p["min_gap_atr"]:
                buy = min(100.0, 50.0 + gap_atr * 50.0)
                regime = "imbalance"
                reason = f"Bullish FVG: gap {gap_atr:.2f} ATR above candle[-3] high"
        elif cur_high < low_2:
            gap_atr = (low_2 - cur_high) / a
            if gap_atr >= p["min_gap_atr"]:
                sell = min(100.0, 50.0 + gap_atr * 50.0)
                regime = "imbalance"
                reason = f"Bearish FVG: gap {gap_atr:.2f} ATR below candle[-3] low"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))