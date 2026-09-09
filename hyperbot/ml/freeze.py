"""Freeze the logistic meta-model into a versioned JSON artifact for deployment.

Trains the logistic once on the cached dataset, extracts the standardizer stats +
logistic coefficients + tercile cutoffs + metadata, writes them to a JSON file, and
SELF-VERIFIES that the pure-formula FrozenModel reproduces the scikit-learn pipeline's
probabilities exactly. After this, nothing retrains on the fly — the live bot, backtest,
and status view all load this one file.

    .venv\\Scripts\\python.exe -m hyperbot.ml.freeze                 # writes models/meta_logistic_v1.json
    .venv\\Scripts\\python.exe -m hyperbot.ml.freeze --version v2    # a new version
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .dataset import CACHE_PATH, FEATURE_COLS, build_dataset
from .model_io import FrozenModel, save_frozen
from .research import _fit_logistic

MODELS_DIR = "models"


def freeze(version: str = "v1", out_path: str | None = None) -> tuple[dict, str]:
    if os.path.exists(CACHE_PATH):
        data = pd.read_csv(CACHE_PATH, parse_dates=["entry_time"])
    else:
        data = build_dataset()
        data.to_csv(CACHE_PATH, index=False)

    X, y = data[FEATURE_COLS].values, data["y"].values
    pipe = _fit_logistic(X, y)
    scaler = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]

    p_train = pipe.predict_proba(X)[:, 1]
    q33, q67 = np.quantile(p_train, [1 / 3, 2 / 3])

    spec = {
        "version": version,
        "model": "logistic",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_window": [str(data["entry_time"].min())[:10], str(data["entry_time"].max())[:10]],
        "n_train": int(len(data)),
        "base_win_rate": round(float(y.mean()), 4),
        "features": list(FEATURE_COLS),
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "terciles": {"q33": float(q33), "q67": float(q67)},
    }

    # Self-verify: the pure-formula FrozenModel must reproduce the sklearn pipeline exactly.
    fm = FrozenModel(spec)
    p_frozen = fm.predict_proba(X)[:, 1]
    max_diff = float(np.max(np.abs(p_frozen - p_train)))
    if max_diff > 1e-9:
        raise SystemExit(f"FROZEN ARTIFACT MISMATCH: max prob diff {max_diff:.2e} (> 1e-9) — not saved")

    out_path = out_path or os.path.join(MODELS_DIR, f"meta_logistic_{version}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_frozen(spec, out_path)
    return spec, out_path, max_diff


def main():
    p = argparse.ArgumentParser(description="Freeze the logistic meta-model to a JSON artifact.")
    p.add_argument("--version", default="v1")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    spec, path, max_diff = freeze(args.version, args.out)
    print(f"frozen {spec['model']} {spec['version']} -> {path}")
    print(f"  trained_at   {spec['trained_at']}")
    print(f"  train window {spec['train_window'][0]} .. {spec['train_window'][1]}  ({spec['n_train']} signals)")
    print(f"  base WR      {spec['base_win_rate']*100:.1f}%")
    print(f"  terciles     q33={spec['terciles']['q33']:.4f}  q67={spec['terciles']['q67']:.4f}")
    print(f"  self-check   pure-formula vs sklearn max diff = {max_diff:.2e}  (OK, < 1e-9)")
    print(f"  top weights  (standardized coef, feature -> pull on P(win)):")
    order = np.argsort(np.abs(spec["coef"]))[::-1][:6]
    for i in order:
        print(f"    {spec['features'][i]:<20} {spec['coef'][i]:>+.3f}")


if __name__ == "__main__":
    main()
