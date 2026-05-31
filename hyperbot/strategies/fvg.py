from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, atr, last_timestamp


class FvgStrategy(Strategy):
    name = "fvg"

    @staticmethod
    def default_params() -> dict:
        return {"atr_period": 14, "proximity_atr": 0.25, "fvg_lookback": 20}

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < max(p["fvg_lookback"], p["atr_period"]) + 3:
            return self.neutral(df, "insufficient data")

        a = float(atr(df, p["atr_period"]).iloc[-1])
        if pd.isna(a) or a <= 0:
            return self.neutral(df, "ATR unavailable")

        high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
        n = len(df)
        c = float(close.iloc[-1])

        # Scan the most recent fvg_lookback bars for the LATEST qualifying FVG.
        start = max(2, n - p["fvg_lookback"])
        latest = None  # (direction, j, zone_bottom, zone_top)
        for j in range(start, n):
            h2 = float(high.iloc[j - 2])
            l2 = float(low.iloc[j - 2])
            lj = float(low.iloc[j])
            hj = float(high.iloc[j])
            if lj > h2:
                latest = ("bull", j, h2, lj)  # zone = (high[j-2], low[j])
            elif hj < l2:
                latest = ("bear", j, hj, l2)  # zone = (high[j], low[j-2])

        if latest is None:
            return StrategySignal(self.name, 0.0, 0.0, "balanced", "No FVG", last_timestamp(df))

        direction, j, zone_bottom, zone_top = latest

        # (2) Unfilled (close-based): no close after j re-entered the zone.
        if direction == "bull":
            filled = any(float(close.iloc[k]) < zone_top for k in range(j + 1, n))
        else:
            filled = any(float(close.iloc[k]) > zone_bottom for k in range(j + 1, n))
        unfilled = not filled

        # (3) Proximity gate: distance from c to nearest zone edge <= proximity_atr*a (0 if inside).
        if c < zone_bottom:
            dist = zone_bottom - c
        elif c > zone_top:
            dist = c - zone_top
        else:
            dist = 0.0
        proximity = dist <= p["proximity_atr"] * a

        # (4) Direction confirmation.
        o = float(open_.iloc[-1])
        if direction == "bull":
            confirm = c > o
        else:
            confirm = c < o

        comps = [True, unfilled, proximity, confirm]
        score = 25.0 * sum(comps)
        if direction == "bull":
            buy, sell = score, 0.0
            label = "buy"
        else:
            buy, sell = 0.0, score
            label = "sell"

        regime = "imbalance"
        reason = (
            f"gap={25 * int(comps[0])} unfilled={25 * int(comps[1])} "
            f"prox={25 * int(comps[2])} confirm={25 * int(comps[3])} "
            f"-> {label} {int(score)}"
        )
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))