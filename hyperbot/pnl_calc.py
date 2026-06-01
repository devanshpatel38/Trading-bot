from __future__ import annotations

import argparse
import json


def compute_stats(result: dict) -> dict:
    trades = result.get("trades", [])
    resolved = [t for t in trades if t.get("outcome") in ("win", "loss")]
    wins = [t for t in resolved if t["outcome"] == "win"]
    n = len(resolved)
    total_r = sum(t["r_multiple"] for t in resolved)
    held = [t["bars_held"] for t in resolved]
    return {
        "trades": len(trades),
        "resolved": n,
        "wins": len(wins),
        "losses": n - len(wins),
        "open": sum(1 for t in trades if t.get("outcome") == "open"),
        "win_rate": round(len(wins) / n * 100, 2) if n else 0.0,
        "total_r": round(total_r, 3),
        "expectancy_r": round(total_r / n, 3) if n else 0.0,
        "avg_bars_held": round(sum(held) / len(held), 2) if held else 0.0,
        "best_r": max((t["r_multiple"] for t in resolved), default=0.0),
        "worst_r": min((t["r_multiple"] for t in resolved), default=0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute PnL stats from backtest JSON.")
    parser.add_argument("results", help="Path to backtest results JSON")
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as fh:
        result = json.load(fh)
    stats = compute_stats(result)
    for key, value in stats.items():
        print(f"{key:>18}: {value}")


if __name__ == "__main__":
    main()