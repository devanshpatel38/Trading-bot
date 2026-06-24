from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from .config import Config
from .data_client import HyperliquidDataClient
from .oi_data import fetch_recent_oi_hourly, regime_series
from .strategies import REGISTRY
from .strategies.base import atr, ema
from .strategies.aggregator import aggregate, aggregate_regime
from .backtest import run_backtest

REGIMES = ("high_fuel", "weak_expansion", "chop", "profit_taking", "bleeding")


def build_regime_rules(cfg) -> dict:
    """Per-regime management dict for run_backtest, from the oi_filter config.
    Only the trade_regime gets the scale-out + breakeven treatment; the rest are plain
    (they are disabled via enabled_regimes anyway)."""
    oi, bc = cfg.oi_filter, cfg.backtest
    plain = {"rr": bc.rr, "partial": None, "breakeven_r": None, "htf": True}
    rules = {k: plain for k in REGIMES}
    rules[oi.trade_regime] = {
        "rr": bc.rr,
        "partial": (oi.partial_frac, oi.partial_r),
        "breakeven_r": oi.breakeven_r,
        "htf": True,
    }
    return rules

STATE_PATH = Path(__file__).parent.parent / "state.json"
NTFY_BASE = "https://ntfy.sh"


def load_state(path: Path = STATE_PATH) -> dict:
    """Returns persisted state dict, or default flat state if file absent."""
    if path.exists():
        return json.loads(path.read_text())
    return {"state": "flat", "trade": None}


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.write_text(json.dumps(state, indent=2))


def send_ntfy(title: str, body: str, topic: str, priority: str = "default") -> None:
    """POST to ntfy.sh. Non-fatal — logs on failure, never raises."""
    try:
        resp = requests.post(
            f"{NTFY_BASE}/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[notifier] ntfy.sh POST failed: {exc}")


def get_current_state(trades: list) -> tuple[str, dict | None, dict | None]:
    """Returns (state, open_trade_or_None, last_resolved_trade_or_None)."""
    open_trades = [t for t in trades if t["outcome"] == "open"]
    resolved = [t for t in trades if t["outcome"] in ("win", "loss")]
    last_resolved = resolved[-1] if resolved else None
    if open_trades:
        return "open", open_trades[-1], last_resolved
    return "flat", None, last_resolved


def check_new_signal(df, strategies: dict, cfg, regime_series=None) -> dict | None:
    """Check if the last closed bar fires a fresh entry, passing the OI-regime gate,
    the unanimous 5/5 vote, and the HTF filter. Returns a trade dict or None.
    Called only when run_backtest shows no open trade (avoids double-entry).

    When the OI filter is enabled, the bar's regime must equal the configured
    trade_regime (chop) and agreement uses aggregate_regime; otherwise it falls back to
    the plain agreement aggregator.
    """
    bc, oi_cfg = cfg.backtest, cfg.oi_filter
    thr = cfg.aggregator.threshold
    sigs = [s.analyze(df) for s in strategies.values()]

    regime = None
    if oi_cfg.enabled and regime_series is not None:
        regime = str(regime_series.iloc[-1])
        if regime != oi_cfg.trade_regime:
            return None
        rec, _ = aggregate_regime(sigs, regime, thr)
    else:
        rec = aggregate(sigs, thr, cfg.aggregator.min_agree, cfg.aggregator.margin).recommendation
    if rec not in ("long", "short"):
        return None

    close = float(df["close"].iloc[-1])
    htf_val = float(ema(df["close"], bc.htf_period).iloc[-1])
    atr_val = float(atr(df, bc.atr_period).iloc[-1])
    if rec == "long" and close <= htf_val:
        return None
    if rec == "short" and close >= htf_val:
        return None
    sign = 1.0 if rec == "long" else -1.0
    stop_dist = bc.atr_mult * atr_val
    stop = close - sign * stop_dist
    tp = close + sign * bc.rr * stop_dist
    trade = {
        "side": rec,
        "entry": round(close, 2),
        "entry_time": str(df.index[-1]),
        "stop": round(stop, 2),
        "tp": round(tp, 2),
        "regime": regime,
    }
    if oi_cfg.enabled and oi_cfg.partial_frac:
        trade["partial"] = round(close + sign * oi_cfg.partial_r * stop_dist, 2)
        trade["partial_frac"] = oi_cfg.partial_frac
    return trade


def notify(prev_state: dict, trades: list, new_signal_trade: dict | None,
           topic: str | None) -> dict:
    """Core transition logic. Sends ntfy notification on state change.
    Returns the new state dict to persist (does NOT save it — caller does).
    """
    curr_state, open_trade, _ = get_current_state(trades)

    if curr_state == "flat" and new_signal_trade is not None:
        curr_state = "new_signal"

    effective_trade = open_trade or new_signal_trade

    if topic:
        if prev_state["state"] == "flat" and curr_state == "new_signal":
            t = new_signal_trade
            side = t["side"].upper()
            emoji = "[LONG]" if t["side"] == "long" else "[SHORT]"
            tag = f" ({t['regime']})" if t.get("regime") else ""
            title = f"BTC {side} Signal{tag}"
            body = f"{emoji} entry {t['entry']:.2f} | SL {t['stop']:.2f} | TP {t['tp']:.2f}"
            if t.get("partial"):
                pct = int(round(t.get("partial_frac", 0.5) * 100))
                body += f" | take {pct}% @ {t['partial']:.2f} (2R)->BE"
            send_ntfy(title, body, topic, priority="high")
            print(f"[notifier] Sent: {title} - {body}")

        elif prev_state["state"] == "open" and curr_state == "flat":
            prev_entry_time = (prev_state.get("trade") or {}).get("entry_time")
            resolved_match = next(
                (t for t in trades
                 if t["outcome"] in ("win", "loss") and t["entry_time"] == prev_entry_time),
                None,
            )
            if resolved_match:
                outcome = resolved_match["outcome"].upper()
                emoji = "[WIN]" if resolved_match["outcome"] == "win" else "[LOSS]"
                title = f"BTC Trade {outcome}"
                r = resolved_match["r_multiple"]
                side = resolved_match["side"].upper()
                body = (f"{emoji} {side} {resolved_match['entry']:.2f} > "
                        f"{resolved_match['exit_price']:.2f} | {r:+.2f}R")
            else:
                prev_trade = prev_state.get("trade") or {}
                title = "BTC Trade Closed"
                side = prev_trade.get("side", "?").upper()
                entry = prev_trade.get("entry", 0.0)
                body = f"{side} {entry:.2f} - outcome unknown"
            send_ntfy(title, body, topic)
            print(f"[notifier] Sent: {title} - {body}")

        else:
            print(f"[notifier] No state change "
                  f"(prev={prev_state['state']}, curr={curr_state}) - silent")

    return {
        "state": "open" if curr_state in ("open", "new_signal") else "flat",
        "trade": effective_trade,
    }


def main() -> None:
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("[notifier] NTFY_TOPIC not set — dry run (state updates, no notifications sent)")

    prev = load_state()

    cfg = Config.load()
    client = HyperliquidDataClient(testnet=False)
    df = client.fetch_candles_days(cfg.symbol, cfg.interval, days=75)
    df = df.iloc[:-1]  # drop forming bar

    strategies = {
        name: REGISTRY[name](scfg.params)
        for name, scfg in cfg.strategies.items()
        if scfg.enabled
    }
    bc = cfg.backtest
    oi_cfg = cfg.oi_filter

    reg = None
    extra = {}
    if oi_cfg.enabled:
        oi = fetch_recent_oi_hourly(oi_cfg.source, days=oi_cfg.recent_days)
        reg = regime_series(df.index, oi, window=oi_cfg.window_hours, avg_hours=oi_cfg.avg_hours)
        extra = dict(regime_series=reg, regime_rules=build_regime_rules(cfg),
                     enabled_regimes={oi_cfg.trade_regime})
        print(f"[notifier] OI filter on — latest regime: {reg.iloc[-1]}")

    trades = run_backtest(
        df, strategies,
        threshold=cfg.aggregator.threshold,
        min_agree=cfg.aggregator.min_agree,
        margin=cfg.aggregator.margin,
        rr=bc.rr, atr_period=bc.atr_period, atr_mult=bc.atr_mult,
        warmup=bc.warmup_bars, fee=bc.fee, slippage=bc.slippage,
        max_window=600, htf_period=bc.htf_period, **extra,
    )

    curr_state_raw, _, _ = get_current_state(trades)
    new_signal_trade = None
    if curr_state_raw == "flat":
        new_signal_trade = check_new_signal(df, strategies, cfg, reg)

    new_state = notify(prev, trades, new_signal_trade, topic)
    save_state(new_state)
    print(f"[notifier] State: {prev['state']} -> {new_state['state']}")


if __name__ == "__main__":
    main()
