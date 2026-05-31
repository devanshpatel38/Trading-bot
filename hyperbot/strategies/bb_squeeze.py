from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, bollinger_bands, last_timestamp


class BbSqueezeStrategy(Strategy):
    name = "bb_squeeze"

    @staticmethod
    def default_params() -> dict:
        return {"period": 20, "num_std": 2.0, "squeeze_lookback": 50, "squeeze_quantile": 0.25}

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["period"] + p["squeeze_lookback"]:
            return self.neutral(df, "insufficient data")
        close = df["close"]
        upper, mid, lower = bollinger_bands(close, p["period"], p["num_std"])
        bandwidth = (upper - lower) / mid
        cur_bw = float(bandwidth.iloc[-1])
        thresh = float(bandwidth.iloc[-p["squeeze_lookback"]:].quantile(p["squeeze_quantile"]))
        squeezing = cur_bw <= thresh
        c, u, l, m = float(close.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1]), float(mid.iloc[-1])
        buy = sell = 0.0
        if c > u:
            buy, regime = 70.0, "trending"
            reason = f"Breakout above upper band (bw {cur_bw:.4f})"
        elif c < l:
            sell, regime = 70.0, "trending"
            reason = f"Breakout below lower band (bw {cur_bw:.4f})"
        elif squeezing:
            regime = "squeeze"
            if c >= m:
                buy = 50.0
                reason = f"BB squeeze (bw {cur_bw:.4f} <= {thresh:.4f}), upward bias"
            else:
                sell = 50.0
                reason = f"BB squeeze (bw {cur_bw:.4f} <= {thresh:.4f}), downward bias"
        else:
            regime = "ranging"
            reason = f"No squeeze/breakout (bw {cur_bw:.4f})"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
