from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, bollinger_bands, last_timestamp


class BbSqueezeStrategy(Strategy):
    name = "bb_squeeze"

    @staticmethod
    def default_params() -> dict:
        return {
            "period": 20,
            "num_std": 2.0,
            "squeeze_lookback": 50,
            "squeeze_quantile": 0.25,
            "vol_lookback": 20,
        }

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["period"] + p["squeeze_lookback"]:
            return self.neutral(df, "insufficient data")

        close = df["close"]
        volume = df["volume"]
        upper, mid, lower = bollinger_bands(close, p["period"], p["num_std"])
        bw = (upper - lower) / mid
        vol_ma = volume.rolling(p["vol_lookback"]).mean()
        bw_thresh = float(bw.iloc[-p["squeeze_lookback"]:].quantile(p["squeeze_quantile"]))

        c = float(close.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(lower.iloc[-1])

        squeeze = float(bw.iloc[-2]) <= bw_thresh
        expansion = float(bw.iloc[-1]) > float(bw.iloc[-2])
        vol_ok = float(volume.iloc[-1]) > float(vol_ma.iloc[-1])
        breakout_up = c > u
        breakout_dn = c < l

        buy_comps = [squeeze, expansion, breakout_up, vol_ok]
        sell_comps = [squeeze, expansion, breakout_dn, vol_ok]
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
                f"breakout={25 * int(buy_comps[2])} vol={25 * int(buy_comps[3])} "
                f"-> buy {int(buy)}"
            )
        else:
            reason = (
                f"squeeze={25 * int(sell_comps[0])} expansion={25 * int(sell_comps[1])} "
                f"breakout={25 * int(sell_comps[2])} vol={25 * int(sell_comps[3])} "
                f"-> sell {int(sell)}"
            )
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))