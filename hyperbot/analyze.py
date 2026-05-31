from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from .config import Config
from .data_client import HyperliquidDataClient
from .strategies import REGISTRY
from .strategies.aggregator import aggregate, AggregatedSignal


def analyze(cfg: Config, df):
    signals, weights = [], {}
    for name, scfg in cfg.strategies.items():
        if not scfg.enabled:
            continue
        signals.append(REGISTRY[name](scfg.params).evaluate(df))
        weights[name] = scfg.weight
    agg = aggregate(
        signals, weights,
        cfg.aggregator.buy_threshold, cfg.aggregator.sell_threshold, cfg.aggregator.margin,
    )
    return signals, agg


def _render(signals, agg: AggregatedSignal, symbol: str, interval: str) -> Table:
    table = Table(title=f"{symbol} {interval} — read-only analysis")
    for col in ["strategy", "buy", "sell", "regime", "reason"]:
        table.add_column(col, overflow="fold")
    for s in signals:
        table.add_row(s.strategy, f"{s.buy_confidence:.0f}", f"{s.sell_confidence:.0f}", s.regime, s.reason)
    table.add_section()
    table.add_row("AGGREGATE", f"{agg.buy_confidence:.0f}", f"{agg.sell_confidence:.0f}", agg.regime, f"DECISION: {agg.decision}")
    return table


def main():
    parser = argparse.ArgumentParser(description="Live read-only market analysis (no orders).")
    parser.add_argument("--config", default="hyperbot/config.yaml")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    client = HyperliquidDataClient(testnet=cfg.testnet)
    df = client.fetch_candles(cfg.symbol, cfg.interval, cfg.lookback)
    signals, agg = analyze(cfg, df)
    Console().print(_render(signals, agg, cfg.symbol, cfg.interval))


if __name__ == "__main__":
    main()
