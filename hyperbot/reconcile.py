"""Real-price forward-test scorecard.

The live bot executes on Binance *Demo Trading* (demo-fapi.binance.com), whose
order book and mark price drift from real mainnet BTC — so a demo win/loss can
disagree with what the strategy actually did on real prices. This module re-runs
the exact chop 4/5 / plain-1:3 engine over **fresh mainnet candles** (fapi) and
scores every trade since the forward-test start against real highs/lows, so the
"status of the trading bot" reflects reality rather than the demo feed.

    python -m hyperbot.reconcile                         # from 2026-07-10 (default)
    python -m hyperbot.reconcile --start 2026-07-10
    python -m hyperbot.reconcile --seed-days 300 --risk 250

Same engine, no lookahead, both-touch bar = loss (conservative) — identical to
backtest.py; the only difference is the trade window is clipped to the live run.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from .config import Config
from .binance_bot import recent_perp
from .oi_data import load_oi_hourly, fetch_live_oi_hourly, regime_series
from .strategies import REGISTRY
from .backtest import run_backtest, REGIME_RULES

FORWARD_START = "2026-07-10"   # first live trade after the OI-fix commit; test counts from here
OI_ATTEMPTS = 5                # retries for the live OI endpoint


def _full_oi(oi_cfg):
    """Full OI archive (deep 30-day references) + live tail, retried.

    The live bot's `hybrid_oi` fetches only a `recent_days` (~50d) archive window, which is
    fine live but breaks a *historical* reconcile: as time passes the window slides forward
    and eventually no longer reaches the 30-day OI reference of the OLDEST forward-test bars,
    so those trades silently vanish from the scorecard. Using the full cached archive (deep
    history) for the references + a 28-day live tail (fresh recent bars, no gap to the cache)
    means every forward-test bar's regime is computable and no trade ever drops.
    """
    archive = load_oi_hourly(oi_cfg.source)                 # full cached history (deep 30d refs)
    for attempt in range(1, OI_ATTEMPTS + 1):
        try:
            live = fetch_live_oi_hourly(oi_cfg.source, days=28)
            return live["oi"].combine_first(archive["oi"]).sort_index().to_frame("oi")
        except Exception:
            if attempt < OI_ATTEMPTS:
                time.sleep(3)
    raise RuntimeError(f"live OI feed unreachable after {OI_ATTEMPTS} attempts — cannot reconcile")


def reconcile(start: str = FORWARD_START, seed_days: int = 300) -> dict:
    """Run the engine on fresh mainnet candles and return resolved + open trades
    whose entry falls on/after `start`, scored on real BTC prices."""
    cfg = Config.load()
    bc, oi = cfg.backtest, cfg.oi_filter
    df = recent_perp(oi.source, cfg.interval, seed_days).iloc[:-1]   # fresh mainnet, drop forming bar
    if oi.enabled:
        oid = _full_oi(oi)                                           # full archive + live tail (no window slide)
        reg = regime_series(df.index, oid, window=oi.window_hours, avg_hours=oi.avg_hours,
                            chop_enter=oi.chop_hyst_enter, chop_exit=oi.chop_hyst_exit)
    else:
        reg = pd.Series("chop", index=df.index)                     # ungated: mirror the live OI-off switch
    strat = {n: REGISTRY[n](s.params) for n, s in cfg.strategies.items() if s.enabled}

    # Apply the frozen meta-model when model_filter is on, so the scorecard reproduces
    # exactly what the live bot did (skip low tercile). Never score on a load failure.
    mdl = cfg.model_filter
    meta = None
    if mdl.enabled:
        from .ml.model_io import load_frozen, make_meta_decider
        fm = load_frozen(mdl.path)
        meta = make_meta_decider(fm, df, htf_period=bc.htf_period, atr_period=bc.atr_period)

    trades = run_backtest(
        df, strat, threshold=cfg.aggregator.threshold, min_agree=cfg.aggregator.min_agree,
        margin=cfg.aggregator.margin, rr=bc.rr, atr_period=bc.atr_period, atr_mult=bc.atr_mult,
        warmup=815, fee=bc.fee, slippage=bc.slippage, htf_period=bc.htf_period, max_window=600,
        regime_series=reg, regime_rules=REGIME_RULES, enabled_regimes={oi.trade_regime},
        chop_min_agree=oi.chop_min_agree, meta=meta,
    )
    resolved = sorted(
        [t for t in trades if t["outcome"] in ("win", "loss") and t["entry_time"][:10] >= start],
        key=lambda t: t["entry_time"],
    )
    open_ = sorted(
        [t for t in trades if t["outcome"] == "open" and t["entry_time"][:10] >= start],
        key=lambda t: t["entry_time"],
    )
    return {"last_bar": str(df.index[-1]), "resolved": resolved, "open": open_,
            "model_on": mdl.enabled}


def _report(start: str, seed_days: int, risk: float) -> None:
    r = reconcile(start, seed_days)
    resolved, open_ = r["resolved"], r["open"]
    print(f"real mainnet candles through {r['last_bar']}")
    mode = "MODEL-gated (skip low tercile)" if r.get("model_on") else "plain ungated"
    print(f"\n=== FORWARD TEST from {start} ({mode}, real-price outcomes) ===")
    wins = losses = 0
    net_r = 0.0
    for i, t in enumerate(resolved, 1):
        wins += t["outcome"] == "win"
        losses += t["outcome"] == "loss"
        net_r += t["r_multiple"]
        tag = f"  [{t['tercile']} P={t['p_win']:.2f}]" if t.get("tercile") else ""
        print(f"{i:>2} {t['entry_time'][:16]} {t['side'].upper():5} "
              f"entry {t['entry']:.1f} SL {t['stop']:.1f} TP {t['tp']:.1f} "
              f"-> {t['r_multiple']:+.2f}R {t['outcome']}{tag}")
    for t in open_:
        tag = f"  [{t['tercile']} P={t['p_win']:.2f}]" if t.get("tercile") else ""
        print(f"    OPEN {t['entry_time'][:16]} {t['side'].upper():5} "
              f"entry {t['entry']:.1f} SL {t['stop']:.1f} TP {t['tp']:.1f} "
              f"(unresolved on real prices){tag}")
    n = wins + losses
    if not n:
        print("\n  no resolved trades in this window yet")
        return
    print(f"\n  {wins}W / {losses}L  ({wins / n * 100:.0f}% win) over {n} resolved trades")
    print(f"  net {net_r:+.2f}R | expectancy {net_r / n:+.3f}R/trade | ~${net_r * risk:,.0f} at ${risk:.0f} risk")


def main() -> None:
    p = argparse.ArgumentParser(description="Real-price forward-test scorecard (mainnet candles).")
    p.add_argument("--start", default=FORWARD_START, help=f"forward-test start date (default {FORWARD_START})")
    p.add_argument("--seed-days", type=int, default=300, help="days of history to seed indicators (EMA800 needs ~34)")
    p.add_argument("--risk", type=float, default=250.0, help="per-trade $ risk for the dollar estimate")
    args = p.parse_args()
    _report(args.start, args.seed_days, args.risk)


if __name__ == "__main__":
    main()