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
    buy_threshold: float
    sell_threshold: float
    margin: float


@dataclass
class BacktestConfig:
    in_sample_bars: int
    out_sample_bars: int
    step: int
    warmup_bars: int
    fee: float
    slippage: float
    risk_fraction: float
    initial_equity: float
    metric: str


@dataclass
class Config:
    symbol: str
    interval: str
    lookback: int
    testnet: bool
    strategies: dict
    aggregator: AggregatorConfig
    backtest: BacktestConfig
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
            api_key=os.getenv("HL_API_KEY"),
            api_secret=os.getenv("HL_API_SECRET"),
        )