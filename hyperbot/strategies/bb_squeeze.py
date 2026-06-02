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
            "squeeze_pct": 0.30,
            "vol_lookback": 20,
        }

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["period"] + p["squeeze_lookback"]:
            return self.neutral(df, "insufficient data")

        close = df["close"]
        open_ = df["open"]
        volume = df["volume"]
        upper, mid, lower = bollinger_bands(close, p["period"], p["num_std"])
        width = (upper - lower) / mid
        vol_avg = volume.rolling(p["vol_lookback"]).mean()

        c = float(close.iloc[-1])
        o = float(open_.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(lower.iloc[-1])
        w = float(width.iloc[-1])
        v = float(volume.iloc[-1])
        va = float(vol_avg.iloc[-1])

        # (1) Squeeze: current BB width in the BOTTOM 30% of its last 50-bar range,
        #     i.e. width[-1] <= the 30th-percentile of the last `squeeze_lookback` widths.
        w_thresh = float(width.iloc[-p["squeeze_lookback"]:].quantile(p["squeeze_pct"]))
        squeeze = w <= w_thresh

        # (2) Breakout (directional).
        breakout_up = c > u
        breakout_dn = c < l

        # (3) Volume: PROPORTIONAL up to 25 (shared, directionless).
        #     pts = 25 * clamp(volume[-1]/vol_avg[-1] - 1.0, 0.0, 1.0)
        #     so volume >= 2x avg -> 25, volume <= avg -> 0; continuous in between.
        if pd.isna(va) or va <= 0:
            vol_pts = 0.0
        else:
            ratio = v / va
            vol_pts = 25.0 * max(0.0, min(1.0, ratio - 1.0))

        # (4) Candle direction.
        candle_up = c > o
        candle_dn = c < o

        squeeze_pts = 25.0 if squeeze else 0.0
        buy = squeeze_pts + (25.0 if breakout_up else 0.0) + vol_pts + (25.0 if candle_up else 0.0)
        sell = squeeze_pts + (25.0 if breakout_dn else 0.0) + vol_pts + (25.0 if candle_dn else 0.0)

        if breakout_up or breakout_dn:
            regime = "expansion"
        elif squeeze:
            regime = "squeeze"
        else:
            regime = "normal"

        if buy >= sell:
            reason = (
                f"squeeze={squeeze_pts:g} breakout={25 if breakout_up else 0} "
                f"volume={vol_pts:g} candle={25 if candle_up else 0} -> buy {buy:g}"
            )
        else:
            reason = (
                f"squeeze={squeeze_pts:g} breakout={25 if breakout_dn else 0} "
                f"volume={vol_pts:g} candle={25 if candle_dn else 0} -> sell {sell:g}"
            )
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
