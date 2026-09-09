"""Model status: what would the meta-model do on the latest closed bar, right now.

Read-only. Fetches fresh BTC perp candles, evaluates the ungated 5-strategy vote on
the last CLOSED bar, and — if it fires — scores it with the logistic meta-model to a
P(win), tercile, and sizing action (skip / mid / high). Also reports the live account
position for context. This is the "signal/status" analogue for the ML layer.

    .venv\\Scripts\\python.exe -m hyperbot.ml.status
    .venv\\Scripts\\python.exe -m hyperbot.ml.status --capital 4742   # size at real equity

NOTE: the model is (re)trained on the cached dataset at load time — deterministic given a
fixed dataset, but a production deploy would load a FROZEN, versioned artifact instead.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ..binance_bot import recent_perp
from ..config import Config
from ..strategies import REGISTRY
from ..strategies.aggregator import aggregate_regime
from ..strategies.base import atr, ema
from .dataset import CACHE_PATH, FEATURE_COLS, build_dataset, feature_row
from .features import market_features
from .model_io import load_frozen
from .research import _fit_logistic

FROZEN_PATH = "models/meta_logistic_v1.json"


def load_model(refit: bool = False):
    """Return (model, q33, q67, n_train, source).

    Default: load the FROZEN deployment artifact (models/meta_logistic_v1.json) — the
    same weights the live bot would run, no retraining. Falls back to training on the
    cached dataset only if no artifact exists or --refit is passed (research mode).
    """
    if not refit and os.path.exists(FROZEN_PATH):
        fm = load_frozen(FROZEN_PATH)
        return fm, fm.q33, fm.q67, fm.n_train, f"frozen {fm.version} ({fm.trained_at})"

    if not os.path.exists(CACHE_PATH):
        data = build_dataset()
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        data.to_csv(CACHE_PATH, index=False)
    else:
        data = pd.read_csv(CACHE_PATH, parse_dates=["entry_time"])
    model = _fit_logistic(data[FEATURE_COLS].values, data["y"].values)
    p = model.predict_proba(data[FEATURE_COLS].values)[:, 1]
    q33, q67 = np.quantile(p, [1 / 3, 2 / 3])
    return model, float(q33), float(q67), len(data), "live-trained (research, NOT frozen)"


def tercile_of(p, q33, q67):
    return "low" if p < q33 else ("mid" if p < q67 else "high")


def account_position(symbol, mainnet):
    try:
        from ..binance_exec import BinanceFuturesClient
        return BinanceFuturesClient(symbol, testnet=not mainnet).position()
    except Exception as exc:
        return {"error": str(exc)}


def main():
    p = argparse.ArgumentParser(description="Meta-model status on the latest closed bar.")
    p.add_argument("--capital", type=float, default=5000.0, help="equity for the $ size (default 5000)")
    p.add_argument("--mainnet", action="store_true", help="check mainnet account (default: demo)")
    p.add_argument("--no-account", action="store_true", help="skip the live account position check")
    p.add_argument("--at", default=None, help="evaluate a specific bar (UTC ts) instead of the latest closed")
    p.add_argument("--refit", action="store_true", help="retrain on the cached dataset instead of the frozen artifact")
    args = p.parse_args()

    cfg = Config.load()
    bc, oi = cfg.backtest, cfg.oi_filter
    model, q33, q67, n_train, source = load_model(refit=args.refit)

    df = recent_perp(oi.source, cfg.interval, days=90).iloc[:-1]  # drop the forming bar
    if args.at:  # evaluate a specific historical bar as if it were the last closed bar
        df = df[df.index <= pd.Timestamp(args.at)]
    bar = str(df.index[-1])
    strategies = {n: REGISTRY[n](s.params) for n, s in cfg.strategies.items() if s.enabled}
    sigs = [s.analyze(df) for s in strategies.values()]
    rec, agreed = aggregate_regime(sigs, "chop", cfg.aggregator.threshold, oi.chop_min_agree)  # ungated

    print("=" * 64)
    print(f"META-MODEL STATUS  {oi.source} {cfg.interval}")
    print(f"last closed bar : {bar}")
    print(f"model           : {source} | {n_train} signals | terciles {q33:.3f} / {q67:.3f}")
    if not args.no_account:
        pos = account_position(oi.source, args.mainnet)
        if pos is None:
            print("account         : FLAT (no open position)")
        elif "error" in pos:
            print(f"account         : (unavailable — {pos['error'][:50]})")
        else:
            print(f"account         : IN POSITION {pos['side']} {abs(pos['qty'])} @ {pos['entry']:.1f} "
                  f"(uPnL {pos['unreal']:+.2f})")
    print("-" * 64)

    if rec not in ("long", "short"):
        agree_buy = sum(1 for s in sigs if s.buy_confidence >= cfg.aggregator.threshold)
        agree_sell = sum(1 for s in sigs if s.sell_confidence >= cfg.aggregator.threshold)
        print(f"ungated vote    : STAND ASIDE (max {max(agree_buy, agree_sell)}/5, "
              f"need {oi.chop_min_agree}/5)")
        print("model           : nothing to score — no signal on this bar")
        print("=" * 64)
        return

    close = float(df["close"].iloc[-1])
    htf = float(ema(df["close"], bc.htf_period).iloc[-1])
    a = float(atr(df, bc.atr_period).iloc[-1])
    htf_ok = (rec == "long" and close > htf) or (rec == "short" and close < htf)
    print(f"ungated vote    : {rec.upper()} ({len(agreed)}/5: {', '.join(agreed)})")
    print(f"HTF filter      : {'PASS' if htf_ok else 'BLOCK'} (close {close:.1f} vs EMA{bc.htf_period} {htf:.1f})")

    if not htf_ok:
        print("model           : not scored - signal blocked by the trend filter (no trade)")
        print("=" * 64)
        return

    # Build the feature vector with the SAME code training uses, then score.
    conf = {s.strategy: {"buy": s.buy_confidence, "sell": s.sell_confidence} for s in sigs}
    mf_row = market_features(df, htf_period=bc.htf_period, atr_period=bc.atr_period).iloc[-1]
    row = feature_row(conf, agreed, rec == "long", mf_row)
    x = np.array([[row[c] for c in FEATURE_COLS]])
    pwin = float(model.predict_proba(x)[0, 1])
    ter = tercile_of(pwin, q33, q67)

    print(f"model P(win)    : {pwin:.3f}  ->  tercile {ter.upper()}")

    if ter == "low":
        print(f"ACTION          : SKIP  (low tercile - model rates this signal poorly)")
        print("=" * 64)
        return

    risk = max(100.0, 0.02 * args.capital) if ter == "mid" else max(250.0, 0.05 * args.capital)
    stop_dist = bc.atr_mult * a
    sign = 1.0 if rec == "long" else -1.0
    stop = close - sign * stop_dist
    tp = close + sign * bc.rr * stop_dist
    qty = risk / stop_dist
    pct = "2%" if ter == "mid" else "5%"
    print(f"ACTION          : TAKE {rec.upper()} @ {ter} sizing ({pct}, floor "
          f"${100 if ter=='mid' else 250:.0f})")
    print(f"  risk          : ${risk:,.0f}  (of ${args.capital:,.0f})  ~= {qty:.4f} BTC")
    print(f"  entry / SL / TP: {close:.1f}  /  {stop:.1f}  /  {tp:.1f}   (RR 1:{bc.rr:.0f}, {bc.atr_mult:.0f}xATR)")
    print("=" * 64)


if __name__ == "__main__":
    main()
