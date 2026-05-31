from hyperbot.config import Config


def test_config_loads_defaults():
    cfg = Config.load("hyperbot/config.yaml")
    assert cfg.symbol == "BTC"
    assert cfg.interval == "15m"
    assert cfg.lookback == 5000
    assert cfg.testnet is True
    assert set(cfg.strategies.keys()) == {
        "ema_trend", "rsi_meanrev", "bb_squeeze", "fvg", "macd_momentum"
    }
    assert cfg.strategies["ema_trend"].grid["fast"] == [13, 21]
    assert cfg.aggregator.buy_threshold == 60
    assert cfg.backtest.warmup_bars == 200
    assert cfg.backtest.metric == "total_return"
