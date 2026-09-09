"""Walk-forward training + evaluation of the meta-model.

For each test year Y: train on every signal BEFORE Y, predict P(win) on year Y
(strictly out-of-sample). Pool the OOS predictions across all test years and ask
the only question that matters: does gating the ungated signals by predicted
P(win) beat taking them all — in expectancy (R/trade) and net R?

Models:
  - LightGBM (workhorse, gradient-boosted trees)
  - Logistic regression (interpretable baseline — must be beaten to justify the GBT)

The gate threshold is chosen on TRAIN data only (max net-R with a min-trade floor)
and applied to the unseen test year, so the reported gated performance is honest.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import CACHE_PATH, FEATURE_COLS, build_dataset


def best_threshold(p: np.ndarray, r: np.ndarray, min_frac: float = 0.25) -> float:
    """Threshold on P(win) that maximizes net R on TRAIN, keeping >= min_frac of trades.

    Chosen only on training data; applied to the unseen test fold. The min-trade
    floor blocks the degenerate 'take one lucky trade' solution.
    """
    n = len(p)
    floor = max(20, int(min_frac * n))
    best_thr, best_val = 0.0, -1e18
    for thr in np.unique(np.quantile(p, np.linspace(0.0, 0.85, 40))):
        mask = p >= thr
        if int(mask.sum()) < floor:
            continue
        val = float(r[mask].sum())
        if val > best_val:
            best_val, best_thr = val, thr
    return best_thr


def _fit_gbt(Xtr, ytr):
    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    spw = (neg / pos) if pos > 0 else 1.0  # balance the ~1:3 win/loss classes
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=4,
        min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=spw, random_state=0, n_jobs=-1, verbosity=-1,
    )
    clf.fit(Xtr, ytr)
    return clf


def _fit_xgb(Xtr, ytr):
    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    spw = (neg / pos) if pos > 0 else 1.0  # balance the ~1:3 win/loss classes
    clf = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, gamma=0.0,
        scale_pos_weight=spw, eval_metric="logloss", tree_method="hist",
        random_state=0, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    return clf


def _fit_logistic(Xtr, ytr):
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    lr.fit(Xtr, ytr)
    return lr


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values("entry_time").reset_index(drop=True)
    data["year"] = pd.to_datetime(data["entry_time"]).dt.year
    years = sorted(data["year"].unique())

    folds = []
    for Y in years:
        train = data[data["year"] < Y]
        test = data[data["year"] == Y]
        if len(train) < 100 or len(test) < 15:
            continue  # not enough history to train, or too few to test
        Xtr, ytr = train[FEATURE_COLS].values, train["y"].values
        Xte = test[FEATURE_COLS].values

        gbt = _fit_gbt(Xtr, ytr)
        xgbm = _fit_xgb(Xtr, ytr)
        lr = _fit_logistic(Xtr, ytr)

        tf = test.copy()
        tf["p_gbt"] = gbt.predict_proba(Xte)[:, 1]
        tf["p_xgb"] = xgbm.predict_proba(Xte)[:, 1]
        tf["p_lr"] = lr.predict_proba(Xte)[:, 1]
        tf["thr_gbt"] = best_threshold(gbt.predict_proba(Xtr)[:, 1], train["r_multiple"].values)
        tf["thr_xgb"] = best_threshold(xgbm.predict_proba(Xtr)[:, 1], train["r_multiple"].values)
        tf["thr_lr"] = best_threshold(lr.predict_proba(Xtr)[:, 1], train["r_multiple"].values)
        folds.append(tf)

    if not folds:
        raise SystemExit("not enough data for any walk-forward fold")
    return pd.concat(folds).reset_index(drop=True)


def _stats(r: np.ndarray) -> dict:
    n = len(r)
    wins = int((r > 0).sum())
    net = float(r.sum())
    return {
        "n": n, "wr": (wins / n * 100 if n else 0.0),
        "net_r": net, "exp": (net / n if n else 0.0),
    }


def _line(label: str, s: dict, risk: float) -> str:
    return (f"  {label:<26} {s['n']:>4} trades | WR {s['wr']:>4.1f}% | "
            f"net {s['net_r']:>+7.2f}R | exp {s['exp']:>+.3f}R/trade | "
            f"~${s['net_r']*risk:>+8,.0f}")


def report(oos: pd.DataFrame, data: pd.DataFrame, risk: float) -> None:
    r = oos["r_multiple"].values
    base = _stats(r)

    take_gbt = oos["p_gbt"] >= oos["thr_gbt"]
    take_xgb = oos["p_xgb"] >= oos["thr_xgb"]
    take_lr = oos["p_lr"] >= oos["thr_lr"]
    gbt = _stats(oos.loc[take_gbt, "r_multiple"].values)
    xgbst = _stats(oos.loc[take_xgb, "r_multiple"].values)
    lr = _stats(oos.loc[take_lr, "r_multiple"].values)

    auc_gbt = roc_auc_score(oos["y"], oos["p_gbt"])
    auc_xgb = roc_auc_score(oos["y"], oos["p_xgb"])
    auc_lr = roc_auc_score(oos["y"], oos["p_lr"])

    print("\n" + "=" * 78)
    print("META-MODEL vs UNGATED BASELINE  —  pooled out-of-sample (walk-forward)")
    print("=" * 78)
    print(f"OOS window: {oos['entry_time'].min()} .. {oos['entry_time'].max()}   "
          f"({len(oos)} test-year signals)\n")
    print(_line("BASELINE  take every signal", base, risk))
    print(_line("GATE  XGBoost P(win)", xgbst, risk))
    print(_line("GATE  LightGBM P(win)", gbt, risk))
    print(_line("GATE  Logistic (baseline)", lr, risk))
    print(f"\n  ranking power (does the model sort winners above losers?)")
    print(f"    OOS AUC   XGBoost {auc_xgb:.3f} | LightGBM {auc_gbt:.3f} | "
          f"Logistic {auc_lr:.3f}   (0.50 = no signal)")

    # Honest separation: expectancy by predicted-probability tercile (no threshold picking)
    print(f"\n  expectancy by XGBoost P(win) tercile (OOS — the 'favorable vs not' view)")
    q = pd.qcut(oos["p_xgb"], 3, labels=["low", "mid", "high"], duplicates="drop")
    for name, grp in oos.groupby(q, observed=True):
        s = _stats(grp["r_multiple"].values)
        print(f"    {str(name):>5}  P~[{grp['p_xgb'].min():.2f},{grp['p_xgb'].max():.2f}]  "
              f"{s['n']:>4} trades | WR {s['wr']:>4.1f}% | exp {s['exp']:>+.3f}R")

    # Per-year, gated (XGBoost) vs baseline
    print(f"\n  per test-year  (baseline exp  ->  XGBoost-gated exp)")
    oy = oos.copy()
    oy["year"] = pd.to_datetime(oy["entry_time"]).dt.year
    for y, grp in oy.groupby("year"):
        b = _stats(grp["r_multiple"].values)
        g = _stats(grp.loc[grp["p_xgb"] >= grp["thr_xgb"], "r_multiple"].values)
        print(f"    {y}   base {b['n']:>3}t {b['exp']:>+.3f}R   ->   "
              f"gated {g['n']:>3}t {g['exp']:>+.3f}R")

    # Which features carry the signal (permutation importance on the final fold's model)
    print(f"\n  top features (permutation importance, final fold)")
    data = data.sort_values("entry_time").reset_index(drop=True)
    data["year"] = pd.to_datetime(data["entry_time"]).dt.year
    last_y = sorted(data["year"].unique())[-1]
    tr, te = data[data["year"] < last_y], data[data["year"] == last_y]
    if len(te) >= 15:
        m = _fit_xgb(tr[FEATURE_COLS].values, tr["y"].values)
        imp = permutation_importance(m, te[FEATURE_COLS].values, te["y"].values,
                                     scoring="roc_auc", n_repeats=8, random_state=0)
        order = np.argsort(imp.importances_mean)[::-1][:8]
        for i in order:
            print(f"    {FEATURE_COLS[i]:<20} {imp.importances_mean[i]:>+.4f}")

    print("\n" + "-" * 78)
    verdict_xgb = xgbst["exp"] - base["exp"]
    print(f"VERDICT: XGBoost gate expectancy {xgbst['exp']:+.3f}R vs baseline {base['exp']:+.3f}R "
          f"= {verdict_xgb:+.3f}R/trade")
    if verdict_xgb > 0.02 and auc_xgb > 0.53:
        print("  -> the meta-model adds edge out-of-sample. Worth taking to a live A/B.")
    elif auc_xgb <= 0.52:
        print("  -> AUC ~0.50: the model can't separate winners from losers on these features. "
              "No edge to gate on.")
    else:
        print("  -> marginal / inconsistent. Not a clear GO; revisit features or reconsider.")
    print("-" * 78)


def main():
    p = argparse.ArgumentParser(description="Walk-forward meta-model research + report.")
    p.add_argument("--rebuild", action="store_true", help="rebuild dataset from candles (else use cache)")
    p.add_argument("--risk", type=float, default=250.0, help="per-trade $ risk for the $ estimate")
    args = p.parse_args()

    if args.rebuild or not os.path.exists(CACHE_PATH):
        print("building dataset (ungated full-history walk of the engine)...")
        data = build_dataset()
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        data.to_csv(CACHE_PATH, index=False)
    else:
        data = pd.read_csv(CACHE_PATH, parse_dates=["entry_time"])

    wins, n = int(data["y"].sum()), len(data)
    print(f"dataset: {n} labeled ungated signals, {data['entry_time'].min().date()} .. "
          f"{data['entry_time'].max().date()} | base WR {wins/n*100:.1f}%")

    oos = walk_forward(data)
    report(oos, data, args.risk)


if __name__ == "__main__":
    main()
