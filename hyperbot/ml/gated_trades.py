"""List the out-of-sample trades the LightGBM meta-gate would TAKE.

Reproduces the exact walk-forward from research.py, keeps only the test-year
signals whose LightGBM P(win) cleared that fold's train-chosen threshold, and
prints them chronologically (also saved to CSV). These are the trades the
ungated strategy fired AND the meta-model approved — i.e. what the filtered
bot would actually have traded.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .dataset import CACHE_PATH, build_dataset
from .research import walk_forward

OUT = "data/ml_gated_trades_lgbm.csv"


def main():
    p = argparse.ArgumentParser(description="List LightGBM-gated OOS trades.")
    p.add_argument("--rebuild", action="store_true", help="rebuild dataset from candles first")
    p.add_argument("--risk", type=float, default=250.0, help="per-trade $ risk for the estimate")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    if args.rebuild or not os.path.exists(CACHE_PATH):
        data = build_dataset()
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        data.to_csv(CACHE_PATH, index=False)
    else:
        data = pd.read_csv(CACHE_PATH, parse_dates=["entry_time"])

    oos = walk_forward(data)
    taken = oos[oos["p_gbt"] >= oos["thr_gbt"]].sort_values("entry_time").reset_index(drop=True)

    cols = ["entry_time", "side", "p_gbt", "thr_gbt", "outcome", "r_multiple"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    taken[cols].to_csv(args.out, index=False)

    print(f"LightGBM-gated OOS trades (P(win) >= per-fold train threshold)")
    print(f"{'#':>3}  {'entry (UTC)':<16} {'side':<5} {'P(win)':>6} {'thr':>5}  {'out':<4} {'R':>7}")
    print("-" * 60)
    wins = 0
    net = 0.0
    for i, t in enumerate(taken.itertuples(index=False), 1):
        wins += t.outcome == "win"
        net += t.r_multiple
        print(f"{i:>3}  {str(t.entry_time)[:16]:<16} {t.side:<5} "
              f"{t.p_gbt:>6.2f} {t.thr_gbt:>5.2f}  {t.outcome:<4} {t.r_multiple:>+7.2f}")
    n = len(taken)
    print("-" * 60)
    print(f"{n} trades | {wins}W / {n-wins}L ({wins/n*100:.1f}% win) | "
          f"net {net:+.2f}R | exp {net/n:+.3f}R | ~${net*args.risk:+,.0f} at ${args.risk:.0f} risk")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
