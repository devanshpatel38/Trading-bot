from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, bollinger_bands, ema, last_timestamp


class BbSqueezeStrategy(Strategy):
    name = "bb_squeeze"

    @staticmethod
    def default_params() -> dict:
        return {
            "period": 20,
            "num_std": 2.0,
            "squeeze_lookback": 50,
            "squeeze_quantile": 0.25,
            "ema_period": 200,
        }

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < max(p["period"] + p["squeeze_lookback"], p["ema_period"] + 1):
            return self.neutral(df, "insufficient data")

        close = df["close"]
        upper, mid, lower = bollinger_bands(close, p["period"], p["num_std"])
        bw = (upper - lower) / mid
        ema200 = ema(close, p["ema_period"])
        bw_thresh = float(bw.iloc[-p["squeeze_lookback"]:].quantile(p["squeeze_quantile"]))

        c = float(close.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(lower.iloc[-1])
        e = float(ema200.iloc[-1])

        squeeze = float(bw.iloc[-2]) <= bw_thresh
        expansion = float(bw.iloc[-1]) > float(bw.iloc[-2])
        breakout_up = c > u
        breakout_dn = c < l
        htf_up = c > e
        htf_dn = c < e

        buy_comps = [squeeze, expansion, breakout_up, htf_up]
        sell_comps = [squeeze, expansion, breakout_dn, htf_dn]
        buy = 25.0 * sum(buy_comps)
        sell = 25.0 * sum(sell_comps)

        if breakout_up or breakout_dn:
            regime = "expansion"
        elif squeeze:
            regime = "squeeze"
        else:
            regime = "ranging"

        if buy >= sell:
            reason = (
                f"squeeze={25 * int(buy_comps[0])} expansion={25 * int(buy_comps[1])} "
                f"breakout={25 * int(buy_comps[2])} htf={25 * int(buy_comps[3])} "
                f"-> buy {int(buy)}"
            )
        else:
            reason = (
                f"squeeze={25 * int(sell_comps[0])} expansion={25 * int(sell_comps[1])} "
                f"breakout={25 * int(sell_comps[2])} htf={25 * int(sell_comps[3])} "
                f"-> sell {int(sell)}"
            )
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))