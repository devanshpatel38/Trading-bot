from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AggregatedSignal:
    recommendation: str  # "long" | "short" | "stand_aside"
    avg_buy: float
    avg_sell: float
    agree_buy: int
    agree_sell: int
    regime: str
    reason: str
    components: list = field(default_factory=list)


def aggregate(signals, threshold: float = 50.0, min_agree: int = 3, margin: float = 15.0) -> AggregatedSignal:
    if not signals:
        return AggregatedSignal("stand_aside", 0.0, 0.0, 0, 0, "unknown", "no signals", [])
    n = len(signals)
    agree_buy = sum(1 for s in signals if s.buy_confidence >= threshold)
    agree_sell = sum(1 for s in signals if s.sell_confidence >= threshold)
    avg_buy = sum(s.buy_confidence for s in signals) / n
    avg_sell = sum(s.sell_confidence for s in signals) / n
    if agree_buy >= min_agree and avg_buy >= threshold * 0.8 and avg_buy > avg_sell + margin:
        rec = "long"
    elif agree_sell >= min_agree and avg_sell >= threshold * 0.8 and avg_sell > avg_buy + margin:
        rec = "short"
    else:
        rec = "stand_aside"
    regime_scores: dict[str, int] = {}
    for s in signals:
        regime_scores[s.regime] = regime_scores.get(s.regime, 0) + 1
    regime = max(regime_scores.items(), key=lambda kv: kv[1])[0]
    reason = (
        f"agree_buy={agree_buy} agree_sell={agree_sell} "
        f"avg_buy={avg_buy:.1f} avg_sell={avg_sell:.1f} -> {rec}"
    )
    return AggregatedSignal(rec, round(avg_buy, 4), round(avg_sell, 4), agree_buy, agree_sell, regime, reason, list(signals))


# Strategy role groups for OI-regime-aware voting.
TREND_STRATS = ("ema_trend", "macd_momentum", "bb_squeeze")
MR_STRATS = ("rsi_meanrev", "fvg")
# Weak-expansion requires the agreeing set to include at least one of these.
CORE_TREND = ("ema_trend", "macd_momentum")


def aggregate_regime(signals, regime: str, threshold: float = 50.0, chop_min_agree: int = 5):
    """OI-regime-aware aggregation. Returns (recommendation, agreed_names).

    recommendation is "long" | "short" | "stand_aside". Voting is purely
    agreement-count based per regime (the count IS the gate); direction is the
    side that meets the regime's required count. Long is checked before short.

      high_fuel      : all 5 agree                          (min_agree=5)
      weak_expansion : >=4 of 5 agree, incl. EMA or MACD    (min_agree=4, conditional)
      chop           : `chop_min_agree` agree               (5=unanimous, 4=relaxed 4/5)
      profit_taking  : all 5 agree                          (min_agree=5)
      bleeding       : both MR strats agree                 (min_agree=2, MR only)
    """
    buys = {s.strategy for s in signals if s.buy_confidence >= threshold}
    sells = {s.strategy for s in signals if s.sell_confidence >= threshold}
    mr, core = set(MR_STRATS), set(CORE_TREND)

    def pick(buy_set, sell_set):
        if buy_set is not None:
            return "long", sorted(buy_set)
        if sell_set is not None:
            return "short", sorted(sell_set)
        return "stand_aside", []

    if regime == "weak_expansion":
        b_ok = len(buys) >= 4 and bool(core & buys)
        s_ok = len(sells) >= 4 and bool(core & sells)
        return pick(buys if b_ok else None, sells if s_ok else None)

    if regime == "bleeding":
        b, s = mr & buys, mr & sells
        return pick(b if len(b) >= 2 else None, s if len(s) >= 2 else None)

    if regime == "chop":
        need = chop_min_agree
        return pick(buys if len(buys) >= need else None, sells if len(sells) >= need else None)

    # high_fuel, profit_taking, and any unknown regime -> strict unanimous 5/5.
    return pick(buys if len(buys) >= 5 else None, sells if len(sells) >= 5 else None)