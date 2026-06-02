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

        c = float(close.iloc[-1])
        e = float(ema200.iloc[-1])
        m = float(macd_line.iloc[-1])
        s = float(signal_line.iloc[-1])
        h1 = float(hist.iloc[-1])
        h0 = float(hist.iloc[-2])

        # (1) HTF: price on the correct side of EMA200.
        htf_buy = c > e
        htf_sell = c < e

        # (2) MACD vs signal line.
        macdsig_buy = m > s
        macdsig_sell = m < s

        # (3) Histogram accelerating (with a sign gate):
        #     buy needs hist > 0 AND rising; sell needs hist < 0 AND falling.
        hist_buy = h1 > 0 and h1 > h0
        hist_sell = h1 < 0 and h1 < h0

        # (4) Fresh cross within cross_lookback bars: a bullish cross is macd crossing
        #     above signal (was <=, now >) on one of the last `cross_lookback` bars.
        lb = p["cross_lookback"]
        cross_buy = any(
            float(macd_line.iloc[-k - 1]) <= float(signal_line.iloc[-k - 1])
            and float(macd_line.iloc[-k]) > float(signal_line.iloc[-k])
            for k in range(1, lb + 1)
        )
        cross_sell = any(
            float(macd_line.iloc[-k - 1]) >= float(signal_line.iloc[-k - 1])
            and float(macd_line.iloc[-k]) < float(signal_line.iloc[-k])
            for k in range(1, lb + 1)
        )

        buy = 25.0 * (int(htf_buy) + int(macdsig_buy) + int(hist_buy) + int(cross_buy))
        sell = 25.0 * (int(htf_sell) + int(macdsig_sell) + int(hist_sell) + int(cross_sell))

        if buy > sell:
            regime = "momentum_up"
        elif sell > buy:
            regime = "momentum_down"
        else:
            regime = "neutral"

        if buy >= sell:
            reason = (
                f"htf={25 if htf_buy else 0} macdsig={25 if macdsig_buy else 0} "
                f"hist={25 if hist_buy else 0} cross={25 if cross_buy else 0} "
                f"-> buy {buy:g}"
            )
        else:
            reason = (
                f"htf={25 if htf_sell else 0} macdsig={25 if macdsig_sell else 0} "
                f"hist={25 if hist_sell else 0} cross={25 if cross_sell else 0} "
                f"-> sell {sell:g}"
            )
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
