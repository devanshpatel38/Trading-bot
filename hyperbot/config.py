from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv


@dataclass
class StrategyConfig:
    enabled: bool
    weight: float
    params: dict
    grid: dict


@dataclass
class AggregatorConfig:
    threshold: float    # confidence required to "agree" (50)
    min_agree: int      # min strategies agreeing (3)
    margin: float       # agreed avg must beat opposite by this (15)


@dataclass
class BacktestConfig:
    days: int
    rr: float
    atr_period: int
    atr_mult: float
    warmup_bars: int
    htf_period: int
    fee: float = 0.0
    slippage: float = 0.0


@dataclass
class OIFilterConfig:
    """OI-regime overlay. When enabled, entries are taken only in `trade_regime`
    (30-day smoothed OI delta), managed with a partial scale-out + breakeven."""
    enabled: bool = False
    source: str = "BTCUSDT"          # Binance USDT-M symbol for OI
    window_hours: int = 720          # 30-day delta lookback
    avg_hours: int = 24              # smoothing window on both endpoints
    trade_regime: str = "chop"       # only enter in this regime
    partial_frac: float = 0.5        # fraction taken at the partial level
    partial_r: float = 2.0           # R level for the partial + breakeven move
    breakeven_r: float = 2.0         # move stop to entry once price reaches this R
    recent_days: int = 50            # live OI fetch window (notifier)


@dataclass
class Config:
    symbol: str
    interval: str
    lookback: int
    testnet: bool
    strategies: dict
    aggregator: AggregatorConfig
    backtest: BacktestConfig
    oi_filter: OIFilterConfig = field(default_factory=OIFilterConfig)
    api_key: str | None = None
    api_secret: str | None = None

    @classmethod
    def load(cls, path: str = "hyperbot/config.yaml", env_path: str = "hyperbot/.env") -> "Config":
        load_dotenv(env_path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        strategies = {
            name: StrategyConfig(
                enabled=bool(s.get("enabled", True)),
                weight=float(s.get("weight", 1.0)),
                params=dict(s.get("params", {})),
                grid=dict(s.get("grid", {})),
            )
            for name, s in raw["strategies"].items()
        }
        return cls(
            symbol=raw["symbol"],
            interval=raw["interval"],
            lookback=int(raw["data"]["lookback"]),
            testnet=bool(raw.get("testnet", True)),
            strategies=strategies,
            aggregator=AggregatorConfig(**raw["aggregator"]),
            backtest=BacktestConfig(**raw["backtest"]),
            oi_filter=OIFilterConfig(**raw.get("oi_filter", {})),
            api_key=os.getenv("HL_API_KEY"),
            api_secret=os.getenv("HL_API_SECRET"),
        )