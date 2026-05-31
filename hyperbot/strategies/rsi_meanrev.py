from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, rsi, ema, atr, last_timestamp


class RsiMeanRevStrategy(Strategy):
    name = "rsi_meanrev"

    @staticmethod
    def default_params() -> dict:
        return {
            "period": 14,
            "oversold": 30.0,
            "overbought": 70.0,
            "ema_period": 50,
            "atr_period": 14,
            "mean_atr": 1.5,
        }

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["ema_period"] + 1:
            return self.neutral(df, "insufficient data")
        close = df["close"]
        r_series = rsi(close, p["period"])
        e = ema(close, p["ema_period"])
        a = atr(df, p["atr_period"])
        r = float(r_series.iloc[-1])
        r_prev = float(r_series.iloc[-2])
        c = float(close.iloc[-1])
        o = float(df["open"].iloc[-1])
        ev = float(e.iloc[-1])
        av = float(a.iloc[-1])
        if pd.isna(r) or pd.isna(r_prev) or pd.isna(av) or av <= 0:
            return self.neutral(df, "indicators unavailable")

        near_mean = abs(c - ev) <= p["mean_atr"] * av

        buy_comps = [r < p["oversold"], r > r_prev, near_mean, c > o]
        sell_comps = [r > p["overbought"], r < r_prev, near_mean, c < o]
        buy = 25.0 * sum(buy_comps)
        sell = 25.0 * sum(sell_comps)
        regime = "ranging"

        if buy >= sell:
            reason = (
                f"rsi={25 * int(buy_comps[0])} turn={25 * int(buy_comps[1])} "
                f"mean={25 * int(buy_comps[2])} candle={25 * int(buy_comps[3])} "
                f"-> buy {int(buy)}"
            )
        else:
            reason = (
                f"rsi={25 * int(sell_comps[0])} turn={25 * int(sell_comps[1])} "
                f"mean={25 * int(sell_comps[2])} candle={25 * int(sell_comps[3])} "
                f"-> sell {int(sell)}"
            )
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
