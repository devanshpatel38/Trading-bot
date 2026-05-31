from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, macd, ema, last_timestamp


class MacdMomentumStrategy(Strategy):
    name = "macd_momentum"

    @staticmethod
    def default_params() -> dict:
        return {"fast": 12, "slow": 26, "signal": 9, "ema_period": 200, "cross_lookback": 3}

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["ema_period"] + 1:
            return self.neutral(df, "insufficient data")

        close = df["close"]
        macd_line, signal_line, hist = macd(close, p["fast"], p["slow"], p["signal"])
        ema200 = ema(close, p["ema_period"])
        m = float(macd_line.iloc[-1])
        c = float(close.iloc[-1])
        e = float(ema200.iloc[-1])

        lb = p["cross_lookback"]
        bullish_cross = any(
            float(macd_line.iloc[-k - 1]) <= float(signal_line.iloc[-k - 1])
            and float(macd_line.iloc[-k]) > float(signal_line.iloc[-k])
            for k in range(1, lb + 1)
        )
        bearish_cross = any(
            float(macd_line.iloc[-k - 1]) >= float(signal_line.iloc[-k - 1])
            and float(macd_line.iloc[-k]) < float(signal_line.iloc[-k])
            for k in range(1, lb + 1)
        )
        hist_up = float(hist.iloc[-1]) > float(hist.iloc[-2])
        hist_dn = float(hist.iloc[-1]) < float(hist.iloc[-2])

        buy_comps = [bullish_cross, hist_up, c > e, m > 0]
        sell_comps = [bearish_cross, hist_dn, c < e, m < 0]
        buy = 25.0 * sum(buy_comps)
        sell = 25.0 * sum(sell_comps)

        if (c > e and bullish_cross) or (c < e and bearish_cross):
            regime = "trending"
        else:
            regime = "ranging"

        if buy >= sell:
            reason = (
                f"cross={25 * int(buy_comps[0])} hist={25 * int(buy_comps[1])} "
                f"htf={25 * int(buy_comps[2])} zero={25 * int(buy_comps[3])} "
                f"-> buy {int(buy)}"
            )
        else:
            reason = (
                f"cross={25 * int(sell_comps[0])} hist={25 * int(sell_comps[1])} "
                f"htf={25 * int(sell_comps[2])} zero={25 * int(sell_comps[3])} "
                f"-> sell {int(sell)}"
            )
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))