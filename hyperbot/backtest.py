from __future__ import annotations

import argparse
import json

import pandas as pd
from rich.console import Console
from rich.table import Table

from .config import Config
from .data_client import HyperliquidDataClient
from .strategies import REGISTRY
from .strategies.base import atr
from .strategies.aggregator import aggregate


def run_backtest(df, strategies, *, threshold, min_agree, margin, rr,
                 atr_period=14, atr_mult=1.5, warmup=215, fee=0.0, slippage=0.0):
    """Walk bars one at a time (no lookahead); one trade at a time."""
    atr_series = atr(df, atr_period)
    closes, highs, lows = df["close"].values, df["high"].values, df["low"].values
    index = df.index
    trades = []
    open_trade = None
    n = len(df)
    for i in range(warmup, n):
        if open_trade is not None:
            hi, lo = float(highs[i]), float(lows[i])
            if open_trade["side"] == "long":
                hit_stop, hit_tp = lo <= open_trade["stop"], hi >= open_trade["tp"]
            else:
                hit_stop, hit_tp = hi >= open_trade["stop"], lo <= open_trade["tp"]
            outcome = None
            if hit_stop:           # conservative: stop wins ties
                outcome = "loss"
            elif hit_tp:
                outcome = "win"
            if outcome is not None:
                exit_price = open_trade["stop"] if outcome == "loss" else open_trade["tp"]
                stop_dist = abs(open_trade["entry"] - open_trade["stop"])
                gross_r = rr if outcome == "win" else -1.0
                cost_r = ((fee + slippage) * (open_trade["entry"] + exit_price) / stop_dist) if stop_dist > 0 else 0.0
                net_r = gross_r - cost_r
                open_trade.update({
                    "outcome": outcome,
                    "exit_time": str(index[i]),
                    "exit_price": exit_price,
                    "bars_held": i - open_trade.pop("_entry_i"),
                    "gross_r": gross_r,
                    "cost_r": round(cost_r, 4),
                    "r_multiple": round(net_r, 4),
                })
                trades.append(open_trade)
                open_trade = None
            continue  # no new signal while managing/closing a trade

        window = df.iloc[: i + 1]  # only past + current bar -> no lookahead
        sigs = [s.analyze(window) for s in strategies.values()]
        agg = aggregate(sigs, threshold, min_agree, margin)
        if agg.recommendation not in ("long", "short"):
            continue
        a = float(atr_series.iloc[i])
        if pd.isna(a) or a <= 0:
            continue
        entry = float(closes[i])
        stop_dist = atr_mult * a
        if agg.recommendation == "long":
            stop, tp = entry - stop_dist, entry + rr * stop_dist
            agreed = [s.strategy for s in sigs if s.buy_confidence >= threshold]
        else:
            stop, tp = entry + stop_dist, entry - rr * stop_dist
            agreed = [s.strategy for s in sigs if s.sell_confidence >= threshold]
        open_trade = {
            "entry_time": str(index[i]), "side": agg.recommendation,
            "entry": entry, "stop": stop, "tp": tp,
            "outcome": None, "exit_time": None, "exit_price": None,
            "bars_held": 0, "gross_r": 0.0, "cost_r": 0.0, "r_multiple": 0.0,
            "strategies_agreed": agreed,
            "confidences": {s.strategy: {"buy": s.buy_confidence, "sell": s.sell_confidence} for s in sigs},
            "_entry_i": i,
        }
    if open_trade is not None:
        open_trade.update({"outcome": "open", "bars_held": n - 1 - open_trade.pop("_entry_i")})
        trades.append(open_trade)
    return trades


def summarize(trades):
    resolved = [t for t in trades if t["outcome"] in ("win", "loss")]
    wins = sum(1 for t in resolved if t["outcome"] == "win")
    total_r = sum(t["r_multiple"] for t in resolved)
    gross_total_r = sum(t["gross_r"] for t in resolved)
    total_cost_r = sum(t["cost_r"] for t in resolved)
    n = len(resolved)
    return {
        "trades": len(trades), "resolved": n, "wins": wins, "losses": n - wins,
        "open": sum(1 for t in trades if t["outcome"] == "open"),
        "win_rate": round(wins / n * 100, 2) if n else 0.0,
        "total_r": round(total_r, 3),
        "expectancy_r": round(total_r / n, 3) if n else 0.0,
        "gross_total_r": round(gross_total_r, 3),
        "total_cost_r": round(total_cost_r, 3),
    }


def attribution(trades, strategy_names):
    rows = {name: {"agreed_wins": 0, "agreed_losses": 0} for name in strategy_names}
    for t in trades:
        if t["outcome"] not in ("win", "loss"):
            continue
        for name in t["strategies_agreed"]:
            rows.setdefault(name, {"agreed_wins": 0, "agreed_losses": 0})
            rows[name]["agreed_wins" if t["outcome"] == "win" else "agreed_losses"] += 1
    for r in rows.values():
        tot = r["agreed_wins"] + r["agreed_losses"]
        r["win_rate_when_agreed"] = round(r["agreed_wins"] / tot * 100, 2) if tot else 0.0
    return rows


def _print_trades(console, trades):
    table = Table(title="Trades")
    for col in ["#", "entry_time", "side", "entry", "stop", "tp", "outcome", "bars", "R", "#agree"]:
        table.add_column(col, overflow="fold")
    for i, t in enumerate(trades, 1):
        table.add_row(str(i), str(t["entry_time"]), t["side"],
                      f"{t['entry']:.2f}", f"{t['stop']:.2f}", f"{t['tp']:.2f}",
                      t["outcome"] or "open", str(t["bars_held"]),
                      f"{t['r_multiple']:+.1f}", str(len(t["strategies_agreed"])))
    console.print(table)


def _print_attribution(console, attr):
    table = Table(title="Strategy attribution (agreement on resolved trades)")
    for col in ["strategy", "agreed_wins", "agreed_losses", "win% when agreed"]:
        table.add_column(col)
    for name, r in attr.items():
        table.add_row(name, str(r["agreed_wins"]), str(r["agreed_losses"]), f"{r['win_rate_when_agreed']:.1f}")
    console.print(table)


def main():
    cfg = Config.load()
    p = argparse.ArgumentParser(description="Event-driven walk-forward backtester (read-only).")
    p.add_argument("--symbol", default=cfg.symbol)
    p.add_argument("--interval", default=cfg.interval)
    p.add_argument("--days", type=int, default=cfg.backtest.days)
    p.add_argument("--rr", type=float, default=cfg.backtest.rr)
    p.add_argument("--confidence", type=float, default=cfg.aggregator.threshold)
    p.add_argument("--minagree", type=int, default=cfg.aggregator.min_agree)
    p.add_argument("--out", default="backtest_results.json")
    args = p.parse_args()

    console = Console()
    client = HyperliquidDataClient(testnet=cfg.testnet)
    df = client.fetch_candles_days(args.symbol, args.interval, args.days)
    console.print(f"Fetched {len(df)} bars for {args.symbol} {args.interval} (~{args.days}d).")

    strategies = {name: REGISTRY[name](scfg.params) for name, scfg in cfg.strategies.items() if scfg.enabled}
    trades = run_backtest(
        df, strategies,
        threshold=args.confidence, min_agree=args.minagree, margin=cfg.aggregator.margin,
        rr=args.rr, atr_period=cfg.backtest.atr_period, atr_mult=cfg.backtest.atr_mult,
        warmup=cfg.backtest.warmup_bars,
        fee=cfg.backtest.fee, slippage=cfg.backtest.slippage,
    )
    summary = summarize(trades)
    attr = attribution(trades, list(strategies.keys()))
    _print_trades(console, trades)
    _print_attribution(console, attr)
    console.print(summary)

    result = {
        "symbol": args.symbol, "interval": args.interval, "days": args.days,
        "rr": args.rr, "confidence": args.confidence, "min_agree": args.minagree,
        "margin": cfg.aggregator.margin, "atr_period": cfg.backtest.atr_period,
        "atr_mult": cfg.backtest.atr_mult, "warmup_bars": cfg.backtest.warmup_bars,
        "fee": cfg.backtest.fee, "slippage": cfg.backtest.slippage,
        "bars": len(df), "trades": trades, "summary": summary, "attribution": attr,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    console.print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()