from hyperbot.strategies import REGISTRY
from hyperbot.strategies.base import StrategySignal
from hyperbot.strategies.aggregator import aggregate, AggregatedSignal


def test_registry_has_all_strategies():
    assert set(REGISTRY.keys()) == {
        "ema_trend", "rsi_meanrev", "bb_squeeze", "fvg", "macd_momentum"
    }


def test_long_when_at_least_three_agree():
    sigs = [
        StrategySignal("a", 80, 0, "trending", "r"),
        StrategySignal("b", 70, 0, "trending", "r"),
        StrategySignal("c", 60, 0, "trending", "r"),
        StrategySignal("d", 10, 0, "ranging", "r"),
        StrategySignal("e", 10, 0, "ranging", "r"),
    ]
    agg = aggregate(sigs, threshold=50, min_agree=3, margin=15)
    assert isinstance(agg, AggregatedSignal)
    assert agg.recommendation == "long"
    assert agg.agree_buy == 3
    assert agg.avg_buy > agg.avg_sell


def test_stand_aside_when_only_two_agree():
    sigs = [
        StrategySignal("a", 80, 0, "trending", "r"),
        StrategySignal("b", 70, 0, "trending", "r"),
        StrategySignal("c", 10, 0, "ranging", "r"),
        StrategySignal("d", 10, 0, "ranging", "r"),
        StrategySignal("e", 10, 0, "ranging", "r"),
    ]
    agg = aggregate(sigs, threshold=50, min_agree=3, margin=15)
    assert agg.agree_buy == 2
    assert agg.recommendation == "stand_aside"


def test_short_when_at_least_three_agree():
    sigs = [
        StrategySignal("a", 0, 80, "trending", "r"),
        StrategySignal("b", 0, 70, "trending", "r"),
        StrategySignal("c", 0, 60, "trending", "r"),
        StrategySignal("d", 0, 10, "ranging", "r"),
        StrategySignal("e", 0, 10, "ranging", "r"),
    ]
    agg = aggregate(sigs, threshold=50, min_agree=3, margin=15)
    assert agg.recommendation == "short"
    assert agg.agree_sell == 3
    assert agg.avg_sell > agg.avg_buy


def test_empty_signals_stand_aside():
    agg = aggregate([], threshold=50, min_agree=3, margin=15)
    assert agg.recommendation == "stand_aside"
    assert agg.avg_buy == 0.0
    assert agg.avg_sell == 0.0