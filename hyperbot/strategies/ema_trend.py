from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, ema, atr, last_timestamp


class EmaTrendStrategy(Strategy):
    name = "ema_trend"

    @staticmethod
    def default_params() -> dict:
        return {"fast": 20, "slow": 200, "atr_period": 14, "pullback_atr": 0.5}

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["slow"] + 1:
            return self.neutral(df, "insufficient data")
        close = df["close"]
        fast = ema(close, p["fast"])
        slow = ema(close, p["slow"])
        a = atr(df, p["atr_period"])
        c = float(close.iloc[-1])
        o = float(df["open"].iloc[-1])
        f = float(fast.iloc[-1])
        s = float(slow.iloc[-1])
        av = float(a.iloc[-1])
        if pd.isna(av) or av <= 0:
            return self.neutral(df, "ATR unavailable")

        in_zone = abs(c - f) <= p["pullback_atr"] * av

        buy_comps = [c > s, f > s, in_zone, c > o]
        sell_comps = [c < s, f < s, in_zone, c < o]
        buy = 25.0 * sum(buy_comps)
        sell = 25.0 * sum(sell_comps)

        if f > s or f < s:
            regime = "trending"
        else:
            regime = "ranging"

        if buy >= sell:
            reason = (
                f"htf={25 * int(buy_comps[0])} align={25 * int(buy_comps[1])} "
                f"pullback={25 * int(buy_comps[2])} candle={25 * int(buy_comps[3])} "
                f"-> buy {int(buy)}"
            )
        else:
            reason = (
                f"htf={25 * int(sell_comps[0])} align={25 * int(sell_comps[1])} "
                f"pullback={25 * int(sell_comps[2])} candle={25 * int(sell_comps[3])} "
                f"-> sell {int(sell)}"
            )
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
