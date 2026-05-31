from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

COLUMNS = ["#", "side", "entry_time", "entry", "exit", "pnl", "buy", "sell", "regime", "reason"]


def render(result: dict) -> Table:
    table = Table(title="Per-trade confidence")
    for col in COLUMNS:
        table.add_column(col, overflow="fold")
    for i, t in enumerate(result.get("trades", []), 1):
        table.add_row(
            str(i),
            t["side"],
            str(t["entry_time"]),
            f"{t['entry_price']:.2f}",
            f"{t.get('exit_price', float('nan')):.2f}",
            f"{t.get('pnl', 0.0):.2f}",
            f"{t['buy_confidence']:.0f}",
            f"{t['sell_confidence']:.0f}",
            t["regime"],
            t["entry_reason"],
        )
    return table


def main():
    parser = argparse.ArgumentParser(description="Render per-trade confidence table.")
    parser.add_argument("results", help="Path to backtest results JSON")
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as fh:
        result = json.load(fh)
    Console().print(render(result))


if __name__ == "__main__":
    main()