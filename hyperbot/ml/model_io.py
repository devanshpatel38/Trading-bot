"""Frozen meta-model artifact: load + score, with NO training dependency.

The deployed model is a plain JSON file (weights, not a pickle) so inference is a
pure, auditable formula that does not depend on the scikit-learn version. A logistic
regression over standardized features is exactly:

    z = intercept + Σ coefᵢ · (xᵢ − meanᵢ) / scaleᵢ
    P(win) = 1 / (1 + e^(−z))

The tercile cutoffs and training metadata travel in the same file, so the live bot,
the backtest, and the status view all load ONE frozen artifact and agree by construction.
"""
from __future__ import annotations

import json

import numpy as np


class FrozenModel:
    """A deployed, frozen logistic meta-model loaded from a JSON spec."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.version = spec["version"]
        self.features = list(spec["features"])
        self.mean = np.asarray(spec["scaler_mean"], dtype=float)
        self.scale = np.asarray(spec["scaler_scale"], dtype=float)
        self.coef = np.asarray(spec["coef"], dtype=float)
        self.intercept = float(spec["intercept"])
        self.q33 = float(spec["terciles"]["q33"])
        self.q67 = float(spec["terciles"]["q67"])
        self.n_train = int(spec.get("n_train", 0))
        self.trained_at = spec.get("trained_at", "?")
        self.train_window = spec.get("train_window", ["?", "?"])

    def _prob(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean) / self.scale
        z = Xs @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X) -> np.ndarray:
        """sklearn-compatible: returns an (n, 2) array of [P(loss), P(win)]."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        p = self._prob(X)
        return np.column_stack([1.0 - p, p])

    def score_row(self, row: dict) -> float:
        """P(win) for one feature dict (keys = self.features)."""
        x = np.array([[row[f] for f in self.features]], dtype=float)
        return float(self._prob(x)[0])

    def tercile(self, p: float) -> str:
        return "low" if p < self.q33 else ("mid" if p < self.q67 else "high")


def make_meta_decider(fm: FrozenModel, df, htf_period: int = 800, atr_period: int = 14):
    """Build a run_backtest `meta` callable from a frozen model.

    Applies the model exactly as the live bot does: score the fired signal, skip the low
    tercile, tag mid/high. Precomputes market_features once over df, then looks up bar i.
    Returns decide(sigs, agreed, is_long, i) -> (take: bool, tercile, p). Imports are local
    so model_io stays dependency-light (numpy only) at import time.
    """
    from .dataset import feature_row
    from .features import market_features

    mf = market_features(df, htf_period=htf_period, atr_period=atr_period)

    def decide(sigs, agreed, is_long, i):
        conf = {s.strategy: {"buy": s.buy_confidence, "sell": s.sell_confidence} for s in sigs}
        p = fm.score_row(feature_row(conf, agreed, is_long, mf.iloc[i]))
        ter = fm.tercile(p)
        return (ter != "low", ter, p)

    return decide


def load_frozen(path: str) -> FrozenModel:
    with open(path, "r", encoding="utf-8") as fh:
        return FrozenModel(json.load(fh))


def save_frozen(spec: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
