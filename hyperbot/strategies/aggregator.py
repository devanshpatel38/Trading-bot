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