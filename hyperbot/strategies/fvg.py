from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, atr, ema, last_timestamp


class FvgStrategy(Strategy):
    name = "fvg"

    @staticmethod
    def default_params() -> dict:
        return {
            "atr_period": 14,
            "ema_period": 200,
            "lookback": 50,
            "fresh_bars": 15,
            "proximity_atr": 0.25,
            "min_gap_atr": 0.08,
        }

    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["ema_period"] + 1:
            return self.neutral(df, "insufficient data")

        a = float(atr(df, p["atr_period"]).iloc[-1])
        if pd.isna(a) or a <= 0:
            return self.neutral(df, "ATR unavailable")

        high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
        ema200 = ema(close, p["ema_period"])
        n = len(df)
        c = float(close.iloc[-1])
        o = float(open_.iloc[-1])
        e = float(ema200.iloc[-1])

        min_gap = p["min_gap_atr"] * a

        # Scan the last `lookback` bars for the LATEST qualifying FVG over triples
        # (j-2, j-1, j). Only count a gap whose size >= min_gap_atr * ATR.
        start = max(2, n - p["lookback"])
        latest = None  # (direction, j, zone_bottom, zone_top)
        for j in range(start, n):
            h2 = float(high.iloc[j - 2])
            l2 = float(low.iloc[j - 2])
            lj = float(low.iloc[j])
            hj = float(high.iloc[j])
            if lj > h2 and (lj - h2) >= min_gap:
                latest = ("bull", j, h2, lj)  # zone = (high[j-2], low[j])
            elif hj < l2 and (l2 - hj) >= min_gap:
                latest = ("bear", j, hj, l2)  # zone = (high[j], low[j-2])

        if latest is None:
            return StrategySignal(self.name, 0.0, 0.0, "balanced", "No FVG", last_timestamp(df))

        direction, j, zone_bottom, zone_top = latest

        # Fill detection (close-based GATE): a filled gap scores nothing.
        if direction == "bull":
            filled = any(float(close.iloc[k]) < zone_top for k in range(j + 1, n))
        else:
            filled = any(float(close.iloc[k]) > zone_bottom for k in range(j + 1, n))
        if filled:
            return StrategySignal(self.name, 0.0, 0.0, "balanced", "FVG filled", last_timestamp(df))

        # (1) Proximity (up to 35, LINEAR). dist = distance from close to nearest zone
        #     edge (0 if inside the zone). The spec says full near the edge and to
        #     "scale down linearly past that": we ramp linearly so that dist=0 -> 35 and
        #     dist >= 2*proximity_atr*a -> 0, using
        #         pts = 35 * clamp(1 - dist/(2*proximity_atr*a), 0, 1).
        if c < zone_bottom:
            dist = zone_bottom - c
        elif c > zone_top:
            dist = c - zone_top
        else:
            dist = 0.0
        span = 2.0 * p["proximity_atr"] * a
        prox_pts = 35.0 * max(0.0, min(1.0, 1.0 - dist / span)) if span > 0 else 0.0

        # (2) Freshness: 25 if gap age (n-1 - j) <= fresh_bars, else 0.
        fresh_pts = 25.0 if ((n - 1) - j) <= p["fresh_bars"] else 0.0

        # (3) HTF: price on the correct side of EMA200.
        if direction == "bull":
            htf = c > e
        else:
            htf = c < e
        htf_pts = 25.0 if htf else 0.0

        # (4) Candle direction.
        if direction == "bull":
            candle = c > o
        else:
            candle = c < o
        candle_pts = 15.0 if candle else 0.0

        score = prox_pts + fresh_pts + htf_pts + candle_pts
        if direction == "bull":
            buy, sell, label = score, 0.0, "buy"
        else:
            buy, sell, label = 0.0, score, "sell"

        reason = (
            f"prox={prox_pts:g} fresh={fresh_pts:g} htf={htf_pts:g} "
            f"candle={candle_pts:g} -> {label} {score:g}"
        )
        return StrategySignal(self.name, buy, sell, "imbalance", reason, last_timestamp(df))
