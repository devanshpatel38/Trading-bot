from __future__ import annotations

import argparse
import json
import math


def compute_stats(result: dict) -> dict:
    trades = result.get("trades", [])
    initial = float(result.get("initial_equity", 0.0))
    final = float(result.get("final_equity", initial))
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / len(trades) * 100.0 if trades else 0.0

    curve = [p["equity"] for p in result.get("equity_curve", [])] or [initial, final]
    peak, max_dd = curve[0], 0.0
    for equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)

    rets = [t["return_pct"] / 100.0 for t in trades]
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var)
        sharpe = mean / std * math.sqrt(len(rets)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round((final - initial) / initial * 100.0, 2) if initial else 0.0,
        "win_rate": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "final_equity": round(final, 2),
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