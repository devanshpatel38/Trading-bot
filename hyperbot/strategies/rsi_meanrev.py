from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, rsi, last_timestamp


class RsiMeanRevStrategy(Strategy):
    name = "rsi_meanrev"

    @staticmethod
    def default_params() -> dict:
        return {"period": 14, "oversold": 30.0, "overbought": 70.0}

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["period"] + 1:
            return self.neutral(df, "insufficient data")
        r = float(rsi(df["close"], p["period"]).iloc[-1])
        if pd.isna(r):
            return self.neutral(df, "RSI unavailable")
        buy = sell = 0.0
        regime = "ranging"
        if r <= p["oversold"]:
            buy = min(100.0, 50.0 + (p["oversold"] - r) / max(p["oversold"], 1e-9) * 50.0)
            reason = f"RSI {r:.1f} oversold (<= {p['oversold']})"
        elif r >= p["overbought"]:
            sell = min(100.0, 50.0 + (r - p["overbought"]) / max(100.0 - p["overbought"], 1e-9) * 50.0)
            reason = f"RSI {r:.1f} overbought (>= {p['overbought']})"
        else:
            reason = f"RSI {r:.1f} neutral"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))