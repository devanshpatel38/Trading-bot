from __future__ import annotations

import argparse

from .config import Config
from .binance_data import load_klines, partition
from .strategies import REGISTRY
from .backtest import run_backtest, summarize


def equity_curve(resolved_trades, risk: float):
    """Compound $100 by net R x risk fraction; return (final_balance, max_drawdown_pct)."""
    bal, peak, maxdd = 100.0, 100.0, 0.0
    for t in resolved_trades:
        bal += t["r_multiple"] * risk * bal
        peak = max(peak, bal)
        maxdd = max(maxdd, (peak - bal) / peak)
    return bal, maxdd * 100.0


def _report(label, res):
    n = len(res)
    if n == 0:
        print(f"  {label}: no resolved trades")
        return n
    wins = sum(1 for t in res if t["outcome"] == "win")
    net = sum(t["r_multiple"] for t in res)
    b1, dd1 = equity_curve(res, 0.01)
    b2, dd2 = equity_curve(res, 0.02)
    print(f"  {label}: trades={n} WR={wins / n * 100:.2f}% netR={net:+.1f} exp={net / n:+.3f} "
          f"| $100@1%=${b1:.2f}({b1 - 100:+.1f}%,DD{dd1:.0f}%) @2%=${b2:.2f}({b2 - 100:+.1f}%,DD{dd2:.0f}%)")
    return n


def main():
    cfg = Config.load()
    A, B = cfg.aggregator, cfg.backtest
    p = argparse.ArgumentParser(description="Out-of-sample validation on Binance spot klines (read-only).")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--tuning-overlap-days", type=int, default=208)
    p.add_argument("--validation-years", type=float, default=2.0)
    args = p.parse_args()

    strats = {n: REGISTRY[n](s.params) for n, s in cfg.strategies.items() if s.enabled}
    print(f"locked config: enabled={list(strats.keys())} min_agree={A.min_agree} "
          f"threshold={A.threshold} rr={B.rr} fee={B.fee} slip={B.slippage}")

    pooled = []
    for sym in [s.strip() for s in args.symbols.split(",")]:
        df = load_klines(sym, args.interval)
        holdout, val, tuning = partition(df, args.tuning_overlap_days, args.validation_years)
        print(f"\n=== {sym} {args.interval} ({len(df)} bars) ===")
        if len(holdout):
            print(f"  RESERVED holdout (NOT analyzed): {holdout.index[0].date()} -> {holdout.index[-1].date()} "
                  f"({len(holdout)} bars)")
        print(f"  validation window: {val.index[0].date()} -> {val.index[-1].date()} ({len(val)} bars)")
        trades = run_backtest(val, strats, threshold=A.threshold, min_agree=A.min_agree, margin=A.margin,
                              rr=B.rr, atr_period=B.atr_period, atr_mult=B.atr_mult,
                              warmup=B.warmup_bars, fee=B.fee, slippage=B.slippage)
        res = [t for t in trades if t["outcome"] in ("win", "loss")]
        pooled.extend(res)
        _report(sym, res)

    print("\n=== POOLED (validation only) ===")
    n = _report("POOLED", pooled)
    print(f"\npooled resolved trades = {n} (target ~400)")


if __name__ == "__main__":
    main()
