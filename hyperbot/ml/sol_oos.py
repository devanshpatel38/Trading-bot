"""Third-asset confirmation: train on POOLED BTC+ETH, test OOS on SOL (unseen).

The BTC->ETH test showed the logistic gate generalizes across one asset boundary.
This is the harder confirmation: pool two assets for training and test on a THIRD
the model has never seen. If the high-P(win) tercile still concentrates the edge on
SOL, the "favorable conditions" signal is genuinely instrument-independent, not a
BTC (or BTC+ETH) artifact.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.metrics import roc_auc_score

from .dataset import CACHE_PATH, FEATURE_COLS, build_dataset
from .eth_oos import ETH_CACHE, FITTERS, LABELS
from .research import _line, _stats, best_threshold

SOL_CACHE = "data/ml_dataset_sol.csv"


def _load(path, builder):
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["entry_time"])
    data = builder()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data.to_csv(path, index=False)
    return data


def main():
    p = argparse.ArgumentParser(description="Train pooled BTC+ETH meta-gate, test OOS on SOL.")
    p.add_argument("--model", choices=list(FITTERS), default="logistic")
    p.add_argument("--rebuild-sol", action="store_true", help="rebuild the SOL dataset from candles")
    p.add_argument("--risk", type=float, default=250.0)
    args = p.parse_args()
    fit, label = FITTERS[args.model], LABELS[args.model]

    btc = _load(CACHE_PATH, build_dataset)
    eth = _load(ETH_CACHE, lambda: build_dataset(symbol="ETHUSDT"))
    if args.rebuild_sol or not os.path.exists(SOL_CACHE):
        print("building SOL ungated dataset (engine walk over SOLUSDT perp history)...")
        sol = build_dataset(symbol="SOLUSDT")
        os.makedirs(os.path.dirname(SOL_CACHE) or ".", exist_ok=True)
        sol.to_csv(SOL_CACHE, index=False)
    else:
        sol = _load(SOL_CACHE, lambda: build_dataset(symbol="SOLUSDT"))

    train = pd.concat([btc, eth], ignore_index=True)
    Xtr, ytr = train[FEATURE_COLS].values, train["y"].values
    model = fit(Xtr, ytr)
    thr = best_threshold(model.predict_proba(Xtr)[:, 1], train["r_multiple"].values)

    p_sol = model.predict_proba(sol[FEATURE_COLS].values)[:, 1]
    take = p_sol >= thr
    base = _stats(sol["r_multiple"].values)
    gated = _stats(sol.loc[take, "r_multiple"].values)
    auc = roc_auc_score(sol["y"], p_sol)

    tw, tn = int(train["y"].sum()), len(train)
    sw, sn = int(sol["y"].sum()), len(sol)

    print("\n" + "=" * 78)
    print(f"THIRD-ASSET OOS  --  {label} trained on POOLED BTC+ETH, tested on SOL (unseen)")
    print("=" * 78)
    print(f"TRAIN  BTC+ETH  {tn} signals   base WR {tw/tn*100:.1f}%   gate threshold (from train) = {thr:.2f}")
    print(f"TEST   SOLUSDT  {sn} signals, {sol['entry_time'].min().date()}..{sol['entry_time'].max().date()}"
          f"   base WR {sw/sn*100:.1f}%\n")
    print(_line("SOL BASELINE take every", base, args.risk))
    print(_line(f"SOL GATED  {label} P(win)", gated, args.risk))
    print(f"\n  OOS AUC on SOL: {auc:.3f}   (0.50 = the pooled model is guessing on SOL)")

    print(f"\n  SOL expectancy by {label} P(win) tercile")
    q = pd.qcut(pd.Series(p_sol), 3, labels=["low", "mid", "high"], duplicates="drop")
    sr = sol["r_multiple"].values
    for name in ["low", "mid", "high"]:
        m = (q == name).values
        if m.sum() == 0:
            continue
        s = _stats(sr[m])
        print(f"    {name:>5}  P~[{p_sol[m].min():.2f},{p_sol[m].max():.2f}]  "
              f"{s['n']:>4} trades | WR {s['wr']:>4.1f}% | exp {s['exp']:>+.3f}R")

    print("\n" + "-" * 78)
    delta = gated["exp"] - base["exp"]
    print(f"VERDICT: SOL gated expectancy {gated['exp']:+.3f}R vs SOL baseline {base['exp']:+.3f}R "
          f"= {delta:+.3f}R/trade  |  SOL AUC {auc:.3f}")
    if delta > 0.03 and auc > 0.53:
        print("  -> confirmed on a THIRD unseen asset. The edge is instrument-independent.")
    elif auc < 0.52 or delta <= 0:
        print("  -> fails on SOL. The BTC+ETH edge does not extend to a third asset.")
    else:
        print("  -> weak/partial on SOL. Suggestive, not decisive.")
    print("-" * 78)


if __name__ == "__main__":
    main()
