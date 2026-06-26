"""Live (demo) execution bot for the plain-1:3 OI-chop strategy on Binance perp.

One idempotent pass per invocation (cron-friendly). Plain 1:3 means SL and TP both
rest on the exchange as closePosition orders, so there is nothing to monitor mid-trade:
  - in a position  -> do nothing (exchange manages SL/TP)
  - flat           -> cancel any leftover bracket, evaluate the chop signal, and if it
                      fires, market-enter + place SL (3xATR) and TP (3R).
Signals + OI come from Binance perp; execution defaults to the demo endpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .config import Config
from .binance_data import fetch_klines, _rows_to_df
from .oi_data import fetch_recent_oi_hourly, regime_series
from .strategies import REGISTRY
from .strategies.base import atr, ema
from .strategies.aggregator import aggregate_regime
from .binance_exec import BinanceFuturesClient

STATE_PATH = Path(__file__).parent.parent / "binance_bot_state.json"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_entry_bar": None}


def save_state(s: dict) -> None:
    STATE_PATH.write_text(json.dumps(s, indent=2))


def recent_perp(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fresh recent perp candles (not cached) for the live signal."""
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    return _rows_to_df(fetch_klines(symbol, interval, start, end, futures=True))


def evaluate_signal(cfg) -> dict | None:
    """Chop signal on the latest CLOSED perp bar. Returns a trade plan or None."""
    oi_cfg, bc = cfg.oi_filter, cfg.backtest
    df = recent_perp(oi_cfg.source, cfg.interval, days=75).iloc[:-1]  # drop forming bar
    oi = fetch_recent_oi_hourly(oi_cfg.source, days=oi_cfg.recent_days)
    reg = regime_series(df.index, oi, window=oi_cfg.window_hours, avg_hours=oi_cfg.avg_hours)
    regime = str(reg.iloc[-1])
    bar = str(df.index[-1])
    if regime != oi_cfg.trade_regime:
        return {"signal": None, "regime": regime, "bar": bar}

    strategies = {n: REGISTRY[n](s.params) for n, s in cfg.strategies.items() if s.enabled}
    sigs = [s.analyze(df) for s in strategies.values()]
    rec, _ = aggregate_regime(sigs, regime, cfg.aggregator.threshold)
    if rec not in ("long", "short"):
        return {"signal": None, "regime": regime, "bar": bar}

    close = float(df["close"].iloc[-1])
    htf = float(ema(df["close"], bc.htf_period).iloc[-1])
    a = float(atr(df, bc.atr_period).iloc[-1])
    if (rec == "long" and close <= htf) or (rec == "short" and close >= htf):
        return {"signal": None, "regime": regime, "bar": bar, "blocked": "htf"}

    sign = 1.0 if rec == "long" else -1.0
    stop_dist = bc.atr_mult * a
    return {"signal": rec, "regime": regime, "bar": bar, "stop_dist": stop_dist,
            "ref_price": close,
            "stop": close - sign * stop_dist, "tp": close + sign * bc.rr * stop_dist}


def run_once(testnet: bool = True, dry: bool = False) -> None:
    cfg = Config.load()
    oi_cfg = cfg.oi_filter
    client = BinanceFuturesClient(oi_cfg.source, testnet=testnet)

    pos = client.position()
    if pos is not None:
        print(f"[bot] in position: {pos['side']} {abs(pos['qty'])} @ {pos['entry']:.1f} "
              f"(mark {pos['mark']:.1f}, uPnL {pos['unreal']:+.2f}) — SL/TP resting, nothing to do")
        return

    # flat: clear any leftover bracket from a just-closed trade
    leftovers = client.open_algo_orders()
    if leftovers:
        print(f"[bot] flat with {len(leftovers)} leftover bracket order(s) -> cancelling")
        if not dry:
            client.cancel_all()

    plan = evaluate_signal(cfg)
    if not plan or plan.get("signal") is None:
        why = plan.get("blocked") or ("regime " + plan.get("regime", "?")) if plan else "?"
        print(f"[bot] stand aside (bar {plan.get('bar') if plan else '?'} | no chop 5/5 | {why})")
        return

    state = load_state()
    if state.get("last_entry_bar") == plan["bar"]:
        print(f"[bot] signal {plan['signal']} but already acted on bar {plan['bar']} — skip")
        return

    bal = client.available_usdt()
    qty = client.round_qty(oi_cfg.risk_pct * bal / plan["stop_dist"])
    side = "BUY" if plan["signal"] == "long" else "SELL"
    close_side = "SELL" if plan["signal"] == "long" else "BUY"
    print(f"[bot] SIGNAL {plan['signal'].upper()} bar {plan['bar']} | ref {plan['ref_price']:.1f} "
          f"qty {qty} (risk ${oi_cfg.risk_pct*bal:.0f}) | SL {client.round_price(plan['stop']):.1f} "
          f"TP {client.round_price(plan['tp']):.1f}")
    if dry:
        print("[bot] DRY RUN — no orders placed")
        return

    o = client.market(side, qty)
    print(f"[bot] entry {o['status']} id {o['orderId']}")
    sl = client.stop_market(close_side, plan["stop"], close_position=True)
    tp = client.take_profit(close_side, plan["tp"], close_position=True)
    print(f"[bot] SL algo {sl.get('algoId')} | TP algo {tp.get('algoId')}")
    state["last_entry_bar"] = plan["bar"]
    save_state(state)
    print("[bot] live (demo). SL + TP resting; exchange will close the position.")


def main():
    p = argparse.ArgumentParser(description="Binance perp chop bot (plain 1:3).")
    p.add_argument("--mainnet", action="store_true", help="trade real mainnet (default: demo)")
    p.add_argument("--dry", action="store_true", help="evaluate + print, place no orders")
    p.add_argument("--loop", type=int, default=0, help="seconds between runs (0 = run once)")
    args = p.parse_args()
    while True:
        try:
            run_once(testnet=not args.mainnet, dry=args.dry)
        except Exception as exc:  # keep the loop alive on transient errors
            print(f"[bot] error: {exc}")
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
