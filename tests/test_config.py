from hyperbot.config import Config


def test_config_loads_defaults():
    cfg = Config.load("hyperbot/config.yaml")
    assert cfg.symbol == "BTC"
    assert cfg.interval == "1h"
    assert cfg.lookback == 5000
    assert cfg.testnet is True
    assert set(cfg.strategies.keys()) == {
        "ema_trend", "rsi_meanrev", "bb_squeeze", "fvg", "macd_momentum"
    }
    # all five enabled for the faithful doc 5/5-unanimous test
    assert all(s.enabled for s in cfg.strategies.values())
    assert cfg.aggregator.threshold == 50
    assert cfg.aggregator.min_agree == 5
    assert cfg.aggregator.margin == 15
    assert cfg.backtest.warmup_bars == 815
    assert cfg.backtest.days == 180
    assert cfg.backtest.rr == 3.0
    assert cfg.backtest.atr_period == 14
    assert cfg.backtest.atr_mult == 3.0
    assert cfg.backtest.htf_period == 800
    assert cfg.backtest.fee == 0.0005   # Binance USDT-M taker 0.05%
    assert cfg.backtest.slippage == 0.0002
    assert cfg.oi_filter.chop_min_agree == 4
    assert cfg.oi_filter.risk_pct == 0.05
    assert cfg.oi_filter.risk_floor == 250
