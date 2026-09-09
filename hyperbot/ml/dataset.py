"""Build the labeled meta-labeling dataset from ungated full-history signals.

Runs the SAME event-driven engine the live bot runs, in UNGATED mode (every bar
is treated as 'chop' so the OI gate is off — take the config's chop_min_agree/5
signal on every bar behind the EMA800 filter, plain 1:3). Each resolved trade
becomes one training row:

    X = signal internals (agreement count, per-strategy directional confidence,
        side) + market context (features.market_features at the entry bar)
    y = 1 if the trade hit TP (win), 0 if it hit stop (loss)

Open/unresolved trades are dropped. The market-context features are precomputed
once as causal series and indexed at each entry bar, so there is no lookahead.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ..backtest import REGIME_RULES, run_backtest
from ..binance_data import load_klines
from ..config import Config
from ..strategies import REGISTRY
from .features import MARKET_COLS, market_features

STRAT_ORDER = ["ema_trend", "rsi_meanrev", "bb_squeeze", "fvg", "macd_momentum"]

FEATURE_COLS = (
    ["agree_count", "side_long"]
    + [f"conf_{s}" for s in STRAT_ORDER]
    + MARKET_COLS
)

CACHE_PATH = "data/ml_dataset.csv"


def feature_row(conf: dict, agreed: list, is_long: bool, mf_row) -> dict:
    """Build the model's feature dict from raw signal + market-context inputs.

    The SINGLE source of truth for the feature vector — used both to build the training
    dataset and (in status.py / any live path) to score a live signal, so live features
    are computed by the exact same code as training. `conf` maps strategy name ->
    {"buy":.., "sell":..}; `agreed` is the list of strategies on the traded side; `mf_row`
    is one row of features.market_features.
    """
    row = {"agree_count": len(agreed), "side_long": 1 if is_long else 0}
    for s in STRAT_ORDER:
        c = conf.get(s, {"buy": 0.0, "sell": 0.0})
        row[f"conf_{s}"] = c["buy"] if is_long else c["sell"]
    for col in MARKET_COLS:
        row[col] = float(mf_row[col])
    return row


def build_dataset(refresh_candles: bool = False, start: str | None = None,
                  symbol: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with FEATURE_COLS + entry_time, side, outcome, r_multiple, y.

    symbol overrides the config's OI source (BTCUSDT) so the same ungated engine can be
    run on another instrument (e.g. ETHUSDT) for a cross-asset out-of-sample test.
    """
    cfg = Config.load()
    bc, oi = cfg.backtest, cfg.oi_filter
    sym = symbol or oi.source
    df = load_klines(sym, cfg.interval, futures=True, refresh=refresh_candles)
    if start:
        df = df[df.index >= pd.Timestamp(start)].copy()

    strategies = {n: REGISTRY[n](s.params) for n, s in cfg.strategies.items() if s.enabled}
    reg = pd.Series("chop", index=df.index)  # UNGATED: every bar eligible, OI gate off
    trades = run_backtest(
        df, strategies,
        threshold=cfg.aggregator.threshold, min_agree=cfg.aggregator.min_agree,
        margin=cfg.aggregator.margin, rr=bc.rr, atr_period=bc.atr_period, atr_mult=bc.atr_mult,
        warmup=bc.warmup_bars, fee=bc.fee, slippage=bc.slippage, htf_period=bc.htf_period,
        max_window=600, regime_series=reg, regime_rules=REGIME_RULES,
        enabled_regimes={oi.trade_regime}, chop_min_agree=oi.chop_min_agree,
    )

    mf = market_features(df, htf_period=bc.htf_period, atr_period=bc.atr_period)

    rows = []
    for t in trades:
        if t["outcome"] not in ("win", "loss"):
            continue
        ts = pd.Timestamp(t["entry_time"])
        if ts not in mf.index:
            continue
        long = t["side"] == "long"
        row = {
            "entry_time": ts,
            "side": t["side"],
            "outcome": t["outcome"],
            "r_multiple": t["r_multiple"],
            "y": 1 if t["outcome"] == "win" else 0,
        }
        row.update(feature_row(t["confidences"], t["strategies_agreed"], long, mf.loc[ts]))
        rows.append(row)

    data = pd.DataFrame(rows).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return data


def main():
    p = argparse.ArgumentParser(description="Build the ungated meta-labeling dataset.")
    p.add_argument("--refresh-candles", action="store_true", help="re-download perp candles")
    p.add_argument("--start", default=None, help="optional start date YYYY-MM-DD")
    p.add_argument("--out", default=CACHE_PATH)
    args = p.parse_args()

    data = build_dataset(refresh_candles=args.refresh_candles, start=args.start)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data.to_csv(args.out, index=False)
    wins = int(data["y"].sum())
    n = len(data)
    print(f"built {n} labeled signals -> {args.out}")
    print(f"  window   {data['entry_time'].min()} .. {data['entry_time'].max()}")
    print(f"  base win rate {wins}/{n} = {wins/n*100:.1f}%   (this is the ungated baseline)")
    print(f"  features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")


if __name__ == "__main__":
    main()
