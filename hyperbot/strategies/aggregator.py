from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AggregatedSignal:
    buy_confidence: float
    sell_confidence: float
    regime: str
    decision: str  # "LONG" | "SHORT" | "FLAT"
    reason: str
    components: list = field(default_factory=list)


def aggregate(signals, weights, buy_threshold, sell_threshold, margin) -> AggregatedSignal:
    total_w = sum(max(0.0, weights.get(s.strategy, 1.0)) for s in signals)
    if not signals or total_w <= 0:
        return AggregatedSignal(0.0, 0.0, "unknown", "FLAT", "no active strategies", [])
    buy = sum(s.buy_confidence * weights.get(s.strategy, 1.0) for s in signals) / total_w
    sell = sum(s.sell_confidence * weights.get(s.strategy, 1.0) for s in signals) / total_w
    regime_scores: dict[str, float] = {}
    for s in signals:
        regime_scores[s.regime] = regime_scores.get(s.regime, 0.0) + weights.get(s.strategy, 1.0)
    regime = max(regime_scores.items(), key=lambda kv: kv[1])[0]
    if buy >= buy_threshold and (buy - sell) >= margin:
        decision = "LONG"
    elif sell >= sell_threshold and (sell - buy) >= margin:
        decision = "SHORT"
    else:
        decision = "FLAT"
    reason = f"buy={buy:.1f} sell={sell:.1f} regime={regime} -> {decision}"
    return AggregatedSignal(round(buy, 4), round(sell, 4), regime, decision, reason, list(signals))
