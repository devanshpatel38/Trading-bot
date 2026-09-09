"""Cross-asset out-of-sample test: train the LightGBM meta-gate on BTC, apply to ETH.

ETH is fully unseen — the model is trained ONLY on BTC ungated signals, and the gate
threshold is chosen ONLY on BTC. We then run the ungated engine on ETH, score every
ETH signal with the BTC-trained model, and compare the gated result to the ungated ETH
baseline. If the gate still lifts ETH expectancy, the learned "favorable conditions"
generalize across instruments; if it collapses to the baseline (or worse), the edge was
BTC-overfit — the exact failure the hand-built chop rule hit on ETH.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .dataset import CACHE_PATH, FEATURE_COLS, build_dataset
from .research import _fit_gbt, _fit_logistic, _fit_xgb, _line, _stats, best_threshold

ETH_CACHE = "data/ml_dataset_eth.csv"

FITTERS = {"logistic": _fit_logistic, "lightgbm": _fit_gbt, "xgboost": _fit_xgb}
LABELS = {"logistic": "Logistic", "lightgbm": "LightGBM", "xgboost": "XGBoost"}


def main():
    p = argparse.ArgumentParser(description="Train a meta-gate on BTC, test OOS on ETH.")
    p.add_argument("--model", choices=list(FITTERS), default="logistic",
                   help="which model to train as the gate (default: logistic)")
    p.add_argument("--rebuild-eth", action="store_true", help="rebuild the ETH dataset from candles")
    p.add_argument("--risk", type=float, default=250.0)
    args = p.parse_args()
    fit = FITTERS[args.model]
    label = LABELS[args.model]

    # BTC training set (reuse the cached full-history dataset)
    if os.path.exists(CACHE_PATH):
        btc = pd.read_csv(CACHE_PATH, parse_dates=["entry_time"])
    else:
        btc = build_dataset()
        btc.to_csv(CACHE_PATH, index=False)

    # ETH test set — fully out-of-sample instrument
    if args.rebuild_eth or not os.path.exists(ETH_CACHE):
        print("building ETH ungated dataset (engine walk over ETHUSDT perp history)...")
        eth = build_dataset(symbol="ETHUSDT")
        os.makedirs(os.path.dirname(ETH_CACHE) or ".", exist_ok=True)
        eth.to_csv(ETH_CACHE, index=False)
    else:
        eth = pd.read_csv(ETH_CACHE, parse_dates=["entry_time"])

    # Train on ALL of BTC; pick the gate threshold on BTC only.
    Xtr, ytr = btc[FEATURE_COLS].values, btc["y"].values
    model = fit(Xtr, ytr)
    thr = best_threshold(model.predict_proba(Xtr)[:, 1], btc["r_multiple"].values)

    p_eth = model.predict_proba(eth[FEATURE_COLS].values)[:, 1]
    take = p_eth >= thr
    base = _stats(eth["r_multiple"].values)
    gated = _stats(eth.loc[take, "r_multiple"].values)
    auc = roc_auc_score(eth["y"], p_eth)

    bw, bn = int(btc["y"].sum()), len(btc)
    ew, en = int(eth["y"].sum()), len(eth)

    print("\n" + "=" * 78)
    print(f"CROSS-ASSET OOS  --  {label} trained on BTC, tested on ETH (unseen instrument)")
    print("=" * 78)
    print(f"TRAIN  BTCUSDT  {bn} signals, {btc['entry_time'].min().date()}..{btc['entry_time'].max().date()}"
          f"  base WR {bw/bn*100:.1f}%   gate threshold (from BTC) = {thr:.2f}")
    print(f"TEST   ETHUSDT  {en} signals, {eth['entry_time'].min().date()}..{eth['entry_time'].max().date()}"
          f"  base WR {ew/en*100:.1f}%\n")
    print(_line("ETH BASELINE take every", base, args.risk))
    print(_line(f"ETH GATED  {label} P(win)", gated, args.risk))
    print(f"\n  OOS AUC on ETH: {auc:.3f}   (0.50 = the BTC model is guessing on ETH)")

    print(f"\n  ETH expectancy by {label} P(win) tercile")
    q = pd.qcut(pd.Series(p_eth), 3, labels=["low", "mid", "high"], duplicates="drop")
    er = eth["r_multiple"].values
    for name in ["low", "mid", "high"]:
        m = (q == name).values
        if m.sum() == 0:
            continue
        s = _stats(er[m])
        print(f"    {name:>5}  P~[{p_eth[m].min():.2f},{p_eth[m].max():.2f}]  "
              f"{s['n']:>4} trades | WR {s['wr']:>4.1f}% | exp {s['exp']:>+.3f}R")

    print("\n" + "-" * 78)
    delta = gated["exp"] - base["exp"]
    print(f"VERDICT: ETH gated expectancy {gated['exp']:+.3f}R vs ETH baseline {base['exp']:+.3f}R "
          f"= {delta:+.3f}R/trade  |  ETH AUC {auc:.3f}")
    if delta > 0.03 and auc > 0.53:
        print("  -> the BTC-trained gate GENERALIZES to ETH. Strong evidence the edge is real.")
    elif auc < 0.52 or delta <= 0:
        print("  -> does NOT generalize to ETH. The edge looks BTC-specific / overfit — "
              "same failure mode as the hand-built chop rule.")
    else:
        print("  -> weak/partial generalization. Not decisive; treat with caution.")
    print("-" * 78)


if __name__ == "__main__":
    main()
