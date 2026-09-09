"""Which model is overfitting? Measure the generalization gap for all three.

Overfitting is invisible in-sample and only shows as the GAP between how a model
scores on data it trained on vs data it never saw. For each model we report three
AUCs on escalating tests:

    train    fit on all BTC, scored on that same BTC   (in-sample — always flatters)
    BTC-OOS  walk-forward, pooled unseen test years    (time out-of-sample)
    ETH-OOS  BTC-trained model scored on ETH           (cross-asset, the harshest)

A model that LEARNED signal keeps these close together and above 0.50. A model that
MEMORIZED noise is high on train and collapses toward 0.50 out-of-sample. The least-
overfit model is the one with the smallest train->OOS drop that still beats 0.50.
"""
from __future__ import annotations

import os

import pandas as pd
from sklearn.metrics import roc_auc_score

from .dataset import CACHE_PATH, FEATURE_COLS, build_dataset
from .eth_oos import ETH_CACHE
from .research import _fit_gbt, _fit_logistic, _fit_xgb, walk_forward

FITTERS = {"XGBoost": _fit_xgb, "LightGBM": _fit_gbt, "Logistic": _fit_logistic}
OOS_COL = {"XGBoost": "p_xgb", "LightGBM": "p_gbt", "Logistic": "p_lr"}


def _load(path, builder):
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["entry_time"])
    data = builder()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data.to_csv(path, index=False)
    return data


def main():
    btc = _load(CACHE_PATH, build_dataset)
    eth = _load(ETH_CACHE, lambda: build_dataset(symbol="ETHUSDT"))

    Xb, yb = btc[FEATURE_COLS].values, btc["y"].values
    Xe, ye = eth[FEATURE_COLS].values, eth["y"].values

    oos = walk_forward(btc)  # pooled time-OOS predictions for all three models

    print("\n" + "=" * 70)
    print("GENERALIZATION GAP  —  the smaller the drop, the less overfit")
    print("=" * 70)
    print(f"{'model':<10} {'train':>7} {'BTC-OOS':>8} {'ETH-OOS':>8}   "
          f"{'drop-OOS':>9} {'drop-ETH':>9}")
    print("-" * 70)

    rows = []
    for name, fit in FITTERS.items():
        model = fit(Xb, yb)
        train_auc = roc_auc_score(yb, model.predict_proba(Xb)[:, 1])
        oos_auc = roc_auc_score(oos["y"], oos[OOS_COL[name]])
        eth_auc = roc_auc_score(ye, model.predict_proba(Xe)[:, 1])
        d_oos = train_auc - oos_auc
        d_eth = train_auc - eth_auc
        rows.append((name, train_auc, oos_auc, eth_auc, d_oos, d_eth))
        print(f"{name:<10} {train_auc:>7.3f} {oos_auc:>8.3f} {eth_auc:>8.3f}   "
              f"{d_oos:>+9.3f} {d_eth:>+9.3f}")

    print("-" * 70)
    print("0.50 = no signal (guessing).  A model 'learning' keeps train ~ OOS ~ ETH > 0.50.")
    print("A big train-to-OOS/ETH drop toward 0.50 = memorizing noise = overfit.\n")

    # least-overfit heuristic: best out-of-sample generalization (highest ETH & OOS,
    # smallest gap). We rank by the worst-case OOS AUC (min of BTC-OOS, ETH-OOS).
    ranked = sorted(rows, key=lambda r: min(r[2], r[3]), reverse=True)
    best = ranked[0]
    print(f"Least-overfit (best worst-case OOS AUC): {best[0]}  "
          f"[BTC-OOS {best[2]:.3f}, ETH-OOS {best[3]:.3f}]")
    if best[3] < 0.52:
        print("  ...but even it can't rank ETH better than a coin flip — on THIS data, none of "
              "the three has a cross-asset edge. 'Least overfit' here means 'least useless'.")


if __name__ == "__main__":
    main()
