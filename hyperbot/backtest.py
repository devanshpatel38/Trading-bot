from __future__ import annotations

import argparse
import itertools
import json
import math

import pandas as pd

from .config import Config
from .data_client import HyperliquidDataClient
from .strategies import REGISTRY
from .strategies.aggregator import aggregate, AggregatedSignal


def make_strategy(name: str, params: dict):
    return REGISTRY[name](params)


def expand_grid(base: dict, grid: dict) -> list[dict]:
    if not grid:
        return [dict(base)]
    keys = list(grid.keys())
    out = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combo = dict(base)
        combo.update(dict(zip(keys, values)))
        out.append(combo)
    return out


def simulate(df, strategies, decide, bt_cfg, start_offset, initial_equity):
    equity = initial_equity
    position = 0  # -1 short, 0 flat, 1 long
    size = 0.0
    open_trade = None
    trades, equity_curve = [], []
    closes = df["close"].values
    index = df.index

    def close_position(exit_price, exit_time, exit_reason):
        nonlocal equity, position, size, open_trade
        gross = position * (exit_price - open_trade["entry_price"]) * size
        fee = bt_cfg["fee"] * abs(size) * (open_trade["entry_price"] + exit_price)
        pnl = gross - fee
        equity += pnl
        notional = open_trade["entry_price"] * size
        open_trade.update({
            "exit_time": str(exit_time),
            "exit_price": exit_price,
            "pnl": pnl,
            "return_pct": (pnl / notional * 100.0) if notional else 0.0,
            "exit_reason": exit_reason,
            "equity_after": equity,
        })
        trades.append(open_trade)
        position, size, open_trade = 0, 0.0, None

    for i in range(start_offset, len(df)):
        window = df.iloc[: i + 1]
        sigs = [s.analyze(window) for s in strategies.values()]
        agg = decide(sigs)
        price = float(closes[i])
        t = index[i]
        target = 1 if agg.recommendation == "long" else (-1 if agg.recommendation == "short" else 0)
        if target != position:
            if position != 0:
                exit_price = price * (1 - position * bt_cfg["slippage"])
                close_position(exit_price, t, agg.reason)
            if target != 0:
                entry_price = price * (1 + target * bt_cfg["slippage"])
                size = (bt_cfg["risk_fraction"] * equity) / entry_price
                position = target
                open_trade = {
                    "side": "LONG" if target == 1 else "SHORT",
                    "entry_time": str(t),
                    "entry_price": entry_price,
                    "size": size,
                    "buy_confidence": agg.avg_buy,
                    "sell_confidence": agg.avg_sell,
                    "regime": agg.regime,
                    "entry_reason": agg.reason,
                }
        equity_curve.append({"time": str(t), "equity": equity})

    if position != 0:
        price = float(closes[-1])
        exit_price = price * (1 - position * bt_cfg["slippage"])
        close_position(exit_price, index[-1], "end of segment")
        equity_curve.append({"time": str(index[-1]), "equity": equity})

    return trades, equity_curve, equity


def metric_value(trades, initial_equity, metric) -> float:
    if not trades:
        return 0.0
    total_pnl = sum(t["pnl"] for t in trades)
    if metric == "sharpe":
        rets = [t["return_pct"] / 100.0 for t in trades]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var)
        return mean / std * math.sqrt(len(rets)) if std > 0 else 0.0
    return total_pnl / initial_equity


def make_aggregator_decider(cfg: Config):
    a = cfg.aggregator
    return lambda sigs: aggregate(sigs, a.threshold, a.min_agree, a.margin)


def make_solo_decider(threshold, margin):
    def decide(sigs):
        s = sigs[0]
        if s.buy_confidence >= threshold and s.buy_confidence > s.sell_confidence + margin:
            rec = "long"
        elif s.sell_confidence >= threshold and s.sell_confidence > s.buy_confidence + margin:
            rec = "short"
        else:
            rec = "stand_aside"
        return AggregatedSignal(rec, s.buy_confidence, s.sell_confidence,
                                int(s.buy_confidence >= threshold), int(s.sell_confidence >= threshold),
                                s.regime, s.reason, [s])
    return decide


def _bt_cfg(cfg: Config) -> dict:
    return {
        "fee": cfg.backtest.fee,
        "slippage": cfg.backtest.slippage,
        "risk_fraction": cfg.backtest.risk_fraction,
    }


def optimize(is_df, cfg: Config) -> dict:
    best = {}
    for name, scfg in cfg.strategies.items():
        if not scfg.enabled:
            continue
        combos = expand_grid(scfg.params, scfg.grid)
        best_metric, best_params = float("-inf"), combos[0]
        for combo in combos:
            strat = make_strategy(name, combo)
            trades, _, _ = simulate(
                is_df, {name: strat},
                make_solo_decider(cfg.aggregator.threshold, cfg.aggregator.margin),
                _bt_cfg(cfg), cfg.backtest.warmup_bars, cfg.backtest.initial_equity,
            )
            m = metric_value(trades, cfg.backtest.initial_equity, cfg.backtest.metric)
            if m > best_metric:
                best_metric, best_params = m, combo
        best[name] = best_params
    return best


def walk_forward(df, cfg: Config) -> dict:
    bt = cfg.backtest
    n = len(df)
    decide = make_aggregator_decider(cfg)
    all_trades, full_curve, windows = [], [], []
    equity = bt.initial_equity
    i = widx = 0
    while i + bt.in_sample_bars + bt.out_sample_bars <= n:
        is_df = df.iloc[i : i + bt.in_sample_bars]
        best = optimize(is_df, cfg)
        oos_start = i + bt.in_sample_bars
        seg_start = max(0, oos_start - bt.warmup_bars)
        seg = df.iloc[seg_start : oos_start + bt.out_sample_bars]
        offset = oos_start - seg_start
        strategies = {name: make_strategy(name, best[name]) for name in best}
        trades, curve, equity = simulate(
            seg, strategies, decide, _bt_cfg(cfg), offset, equity
        )
        all_trades.extend(trades)
        full_curve.extend(curve)
        windows.append({
            "index": widx,
            "in_sample_start": int(i),
            "oos_start": int(oos_start),
            "params": best,
            "trades": len(trades),
            "end_equity": equity,
        })
        i += bt.step
        widx += 1
    return {
        "trades": all_trades,
        "equity_curve": full_curve,
        "windows": windows,
        "initial_equity": bt.initial_equity,
        "final_equity": equity,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest (read-only).")
    parser.add_argument("--config", default="hyperbot/config.yaml")
    parser.add_argument("--out", default="backtest_results.json")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    client = HyperliquidDataClient(testnet=cfg.testnet)
    df = client.fetch_candles(cfg.symbol, cfg.interval, cfg.lookback)
    result = walk_forward(df, cfg)
    result["symbol"] = cfg.symbol
    result["interval"] = cfg.interval
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"Wrote {len(result['trades'])} trades, final equity {result['final_equity']:.2f} -> {args.out}")


if __name__ == "__main__":
    main()