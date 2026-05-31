import numpy as np
import pandas as pd

from hyperbot.config import (
    Config, StrategyConfig, AggregatorConfig, BacktestConfig
)
from hyperbot.backtest import expand_grid, simulate, metric_value, walk_forward
from hyperbot.strategies.base import Strategy, StrategySignal


def _ramp_df(n):
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    close = np.linspace(100, 300, n)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


class _AlwaysLong(Strategy):
    name = "always_long"

    def analyze(self, df):
        return StrategySignal(self.name, 100.0, 0.0, "trending", "stub", df.index[-1])


def test_expand_grid_cartesian_product():
    combos = expand_grid({"a": 1, "b": 2}, {"a": [1, 2], "b": [3]})
    assert len(combos) == 2
    assert {"a": 1, "b": 3} in combos
    assert {"a": 2, "b": 3} in combos


def test_expand_grid_empty_returns_base():
    assert expand_grid({"a": 1}, {}) == [{"a": 1}]


def test_simulate_long_in_uptrend_profits():
    df = _ramp_df(50)
    agg_cfg = {"buy_threshold": 60, "sell_threshold": 60, "margin": 10}
    bt_cfg = {"fee": 0.0, "slippage": 0.0, "risk_fraction": 1.0}
    trades, curve, equity = simulate(
        df, {"always_long": _AlwaysLong()}, {"always_long": 1.0},
        agg_cfg, bt_cfg, start_offset=5, initial_equity=10000.0,
    )
    assert equity > 10000.0
    assert len(trades) >= 1
    assert trades[0]["side"] == "LONG"


def test_metric_value_total_return():
    trades = [{"pnl": 500.0, "return_pct": 5.0}, {"pnl": -100.0, "return_pct": -1.0}]
    assert metric_value(trades, 10000.0, "total_return") == 0.04


def test_walk_forward_smoke():
    df = _ramp_df(100)
    cfg = Config(
        symbol="BTC", interval="15m", lookback=100, testnet=True,
        strategies={
            "ema_trend": StrategyConfig(
                enabled=True, weight=1.0,
                params={"fast": 5, "slow": 10, "atr_period": 5, "pullback_atr": 2.0, "trend_min_pct": 0.0},
                grid={"fast": [5, 8]},
            )
        },
        aggregator=AggregatorConfig(buy_threshold=40, sell_threshold=40, margin=5),
        backtest=BacktestConfig(
            in_sample_bars=40, out_sample_bars=20, step=20, warmup_bars=15,
            fee=0.0005, slippage=0.0002, risk_fraction=0.1,
            initial_equity=10000.0, metric="total_return",
        ),
    )
    result = walk_forward(df, cfg)
    assert "trades" in result and "equity_curve" in result and "windows" in result
    assert len(result["windows"]) >= 1
    assert result["windows"][0]["params"]["ema_trend"]["fast"] in (5, 8)