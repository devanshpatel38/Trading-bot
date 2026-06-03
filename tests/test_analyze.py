import numpy as np
import pandas as pd

from hyperbot.config import Config, StrategyConfig, AggregatorConfig, BacktestConfig
from hyperbot.analyze import analyze
from hyperbot.strategies.aggregator import AggregatedSignal


def _cfg():
    return Config(
        symbol="BTC", interval="15m", lookback=200, testnet=True,
        strategies={
            "ema_trend": StrategyConfig(True, 1.0, {"fast": 5, "slow": 10}, {}),
            "rsi_meanrev": StrategyConfig(True, 1.0, {"period": 14}, {}),
        },
        aggregator=AggregatorConfig(50, 3, 15),
        backtest=BacktestConfig(days=30, rr=2.0, atr_period=14, atr_mult=1.5, warmup_bars=15, htf_period=800),
    )


def test_analyze_returns_signals_and_aggregate():
    idx = pd.date_range("2021-01-01", periods=120, freq="15min")
    close = pd.Series(np.linspace(100, 200, 120), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )
    sigs, agg = analyze(_cfg(), df)
    assert len(sigs) == 2
    assert isinstance(agg, AggregatedSignal)
    assert agg.recommendation in {"long", "short", "stand_aside"}