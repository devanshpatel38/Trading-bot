from hyperbot.strategies import REGISTRY
from hyperbot.strategies.base import StrategySignal
from hyperbot.strategies.aggregator import aggregate, AggregatedSignal


def test_registry_has_all_strategies():
    assert set(REGISTRY.keys()) == {
        "ema_trend", "rsi_meanrev", "bb_squeeze", "fvg", "macd_momentum"
    }


def test_aggregate_weighted_mean_and_long_decision():
    sigs = [
        StrategySignal("a", 80, 0, "trending", "r1"),
        StrategySignal("b", 40, 0, "trending", "r2"),
    ]
    agg = aggregate(sigs, {"a": 1.0, "b": 1.0}, buy_threshold=60, sell_threshold=60, margin=10)
    assert isinstance(agg, AggregatedSignal)
    assert agg.buy_confidence == 60.0
    assert agg.sell_confidence == 0.0
    assert agg.regime == "trending"
    assert agg.decision == "LONG"


def test_aggregate_flat_when_below_threshold():
    sigs = [StrategySignal("a", 50, 0, "ranging", "r")]
    agg = aggregate(sigs, {"a": 1.0}, buy_threshold=60, sell_threshold=60, margin=10)
    assert agg.decision == "FLAT"


def test_aggregate_short_decision():
    sigs = [StrategySignal("a", 0, 90, "trending", "r")]
    agg = aggregate(sigs, {"a": 1.0}, buy_threshold=60, sell_threshold=60, margin=10)
    assert agg.decision == "SHORT"
