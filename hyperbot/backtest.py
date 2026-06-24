from __future__ import annotations

import argparse
import json

import pandas as pd
from rich.console import Console
from rich.table import Table

from .config import Config
from .data_client import HyperliquidDataClient
from .strategies import REGISTRY
from .strategies.base import atr, ema
from .strategies.aggregator import aggregate, aggregate_regime


# Per-regime trade management. rr = TP target (R). partial = (fraction, level_R)
# scaled out early. breakeven_r = move stop to entry once price reaches +Nx risk.
# htf = whether the EMA(htf_period) trend filter gates entries in this regime.
# weak_expansion & profit_taking share management: scale 50% out at 2R, move stop to
# breakeven at that same 2R level, let the runner go to 3R.
REGIME_RULES = {
    "high_fuel":      {"rr": 3.0, "partial": None,       "breakeven_r": None, "htf": True},
    "weak_expansion": {"rr": 3.0, "partial": (0.5, 2.0), "breakeven_r": 2.0,  "htf": True},
    "chop":           {"rr": 3.0, "partial": None,       "breakeven_r": None, "htf": True},
    "profit_taking":  {"rr": 3.0, "partial": (0.5, 2.0), "breakeven_r": 2.0,  "htf": True},
    "bleeding":       {"rr": 3.0, "partial": None,       "breakeven_r": None, "htf": False},
}


def _close_trade(ot, exit_price, i, index, fee, slippage):
    """Finalize an open trade at exit_price, computing R across partial + runner legs.

    Reduces exactly to the single-leg model (gross +rr / -1, cost on entry+exit) when
    the trade never scaled and never moved to breakeven.
    """
    entry, sd, rr_t = ot["entry"], ot["stop_dist"], ot["rr"]
    fp = ot["partial_frac"] if ot["scaled"] else 0.0
    fr = 1.0 - fp
    if exit_price == ot["tp"]:
        runner_r = rr_t
    elif ot["be_moved"] and exit_price == entry:
        runner_r = 0.0
    else:  # original stop
        runner_r = -1.0
    partial_r = ot["partial_level_r"] if ot["scaled"] else 0.0
    gross_r = fp * partial_r + fr * runner_r

    fs = fee + slippage
    if sd > 0:
        cost_r = fs * entry / sd                       # entry, full size
        cost_r += fs * exit_price / sd * fr            # runner exit
        if ot["scaled"]:
            cost_r += fs * ot["partial_exit"] / sd * fp  # partial exit
    else:
        cost_r = 0.0
    net_r = gross_r - cost_r
    ot.update({
        "outcome": "win" if net_r > 0 else "loss",
        "exit_time": str(index[i]),
        "exit_price": exit_price,
        "bars_held": i - ot.pop("_entry_i"),
        "gross_r": round(gross_r, 4),
        "cost_r": round(cost_r, 4),
        "r_multiple": round(net_r, 4),
    })


def run_backtest(df, strategies, *, threshold, min_agree, margin, rr,
                 atr_period=14, atr_mult=1.5, warmup=215, fee=0.0, slippage=0.0,
                 one_per_day=False, max_window=None, htf_period=None,
                 regime_series=None, regime_rules=None, enabled_regimes=None):
    """Walk bars one at a time (no lookahead); one trade at a time.

    one_per_day: if True, take at most one entry per calendar day.
    max_window: if set, feed strategies only the trailing `max_window` bars instead of
        the full history. All indicators look back <= ~200 bars, so a window comfortably
        above that (e.g. 600) yields effectively identical signals while turning the
        bar-by-bar scan from O(n^2) into O(n) — needed for large (15m) datasets.
    regime_series: optional str Series aligned to df.index giving the OI regime per bar.
        When provided, entries use aggregate_regime() and management uses REGIME_RULES
        (per-regime RR, partial scale-out, breakeven, HTF toggle). The regime is fixed at
        entry and governs the trade for its whole life. When None, behaviour is identical
        to the agreement-based aggregator with a single fixed rr/stop/tp.
    """
    rules_map = regime_rules or REGIME_RULES
    atr_series = atr(df, atr_period)
    htf_ema = ema(df["close"], htf_period) if htf_period else None
    closes, highs, lows = df["close"].values, df["high"].values, df["low"].values
    index = df.index
    trades = []
    open_trade = None
    last_entry_date = None
    n = len(df)
    for i in range(warmup, n):
        if open_trade is not None:
            ot = open_trade
            hi, lo = float(highs[i]), float(lows[i])
            if ot["side"] == "long":
                hit_stop = lo <= ot["stop"]
                hit_tp = hi >= ot["tp"]
                hit_partial = ot["partial_price"] is not None and not ot["scaled"] and hi >= ot["partial_price"]
                hit_be = ot["be_price"] is not None and not ot["be_moved"] and hi >= ot["be_price"]
            else:
                hit_stop = hi >= ot["stop"]
                hit_tp = lo <= ot["tp"]
                hit_partial = ot["partial_price"] is not None and not ot["scaled"] and lo <= ot["partial_price"]
                hit_be = ot["be_price"] is not None and not ot["be_moved"] and lo <= ot["be_price"]

            if hit_stop:                       # conservative: stop wins all ties
                _close_trade(ot, ot["stop"], i, index, fee, slippage)
                trades.append(ot)
                open_trade = None
            elif hit_tp:
                if hit_partial:                # bank the partial leg before the runner exits
                    ot["scaled"] = True
                    ot["partial_exit"] = ot["partial_price"]
                _close_trade(ot, ot["tp"], i, index, fee, slippage)
                trades.append(ot)
                open_trade = None
            else:
                if hit_partial:
                    ot["scaled"] = True
                    ot["partial_exit"] = ot["partial_price"]
                if hit_be:
                    ot["stop"] = ot["entry"]
                    ot["be_moved"] = True
            continue  # no new signal while managing/closing a trade

        regime = None
        if regime_series is not None:
            regime = str(regime_series.iloc[i])
            if regime == "unknown":
                continue  # no valid OI history -> never trade
            if enabled_regimes is not None and regime not in enabled_regimes:
                continue  # regime disabled -> stand aside (also skips strategy compute)

        w_start = 0 if max_window is None else max(0, i - max_window + 1)
        window = df.iloc[w_start: i + 1]  # only past + current bar -> no lookahead
        sigs = [s.analyze(window) for s in strategies.values()]

        if regime_series is not None:
            rules = rules_map.get(regime, rules_map["chop"])
            rec, agreed = aggregate_regime(sigs, regime, threshold)
            rr_t = rules["rr"]
            partial = rules["partial"]
            be_r = rules["breakeven_r"]
            htf_on = rules["htf"]
        else:
            regime = None
            agg = aggregate(sigs, threshold, min_agree, margin)
            rec = agg.recommendation
            rr_t, partial, be_r, htf_on = rr, None, None, True
            if rec == "long":
                agreed = [s.strategy for s in sigs if s.buy_confidence >= threshold]
            elif rec == "short":
                agreed = [s.strategy for s in sigs if s.sell_confidence >= threshold]
            else:
                agreed = []

        if rec not in ("long", "short"):
            continue
        a = float(atr_series.iloc[i])
        if pd.isna(a) or a <= 0:
            continue
        bar_date = index[i].date()
        if one_per_day and bar_date == last_entry_date:
            continue  # already entered a trade today
        entry = float(closes[i])
        if htf_period is not None and htf_on:
            h = float(htf_ema.iloc[i])
            if rec == "long" and not (entry > h):
                continue
            if rec == "short" and not (entry < h):
                continue
        last_entry_date = bar_date
        stop_dist = atr_mult * a
        sign = 1.0 if rec == "long" else -1.0
        stop = entry - sign * stop_dist
        tp = entry + sign * rr_t * stop_dist
        partial_price = entry + sign * partial[1] * stop_dist if partial else None
        be_price = entry + sign * be_r * stop_dist if be_r else None
        open_trade = {
            "entry_time": str(index[i]), "side": rec,
            "entry": entry, "stop": stop, "tp": tp, "rr": rr_t,
            "regime": regime, "stop_dist": stop_dist,
            "partial_price": partial_price, "partial_frac": partial[0] if partial else 0.0,
            "partial_level_r": partial[1] if partial else 0.0,
            "scaled": False, "partial_exit": None,
            "be_price": be_price, "be_moved": False,
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
        htf_period=cfg.backtest.htf_period,
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