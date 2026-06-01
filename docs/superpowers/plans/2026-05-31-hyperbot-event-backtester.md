# Hyperbot — Event-driven walk-forward backtester (PDF p.11)

> Executed via superpowers:subagent-driven-development. Each STEP ends with the full suite green (`.venv\Scripts\python.exe -m pytest -q`). Plain `git commit` (local identity set). Do NOT push (controller pushes at checkpoints).

**Goal:** Replace the grid-optimization `backtest.py` with the doc's event-driven walk-forward simulator: scan bars one at a time (no lookahead), enter at the close of a signal bar, manage a fixed `1.5×ATR(14)` stop and `rr×stop` TP one trade at a time, record rich per-trade data + strategy attribution, print tables, and save `backtest_results.json`. Accounting is in R-multiples.

**Confirmed decisions:** replace `backtest.py`; backtester is self-contained (prints its own tables); R-multiples (+rr on TP, -1 on stop, 0/excluded if still open); retire `show_signals.py`.

**Doc-exact parameters:** warmup 215; stop 1.5×ATR(14); TP = rr×stop with rr default 2.0 (1:2); both SL+TP in one bar → **loss**; `--days` up to 180; pagination moves end timestamp backward (~5000/call cap).

---

## STEP 1 — Paginated day-based candle fetch + config changes

### 1a. `data_client.py` — add `fetch_candles_days`
Add (keep existing `fetch_candles` and `INTERVAL_MS`):
```python
    def fetch_candles_days(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch `days` of candles, paginating backward (API caps ~5000/call)."""
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported interval '{interval}'. Choose from {list(INTERVAL_MS)}")
        step = INTERVAL_MS[interval]
        end = int(time.time() * 1000)
        start = end - days * 86_400_000
        rows: dict[int, dict] = {}
        cursor_end = end
        while cursor_end > start:
            batch = self.info.candle_snapshot(symbol, interval, start, cursor_end)
            if not batch:
                break
            for c in batch:
                rows[int(c["t"])] = c
            earliest = min(int(c["t"]) for c in batch)
            if earliest <= start or earliest >= cursor_end:
                break  # covered the window, or no backward progress
            cursor_end = earliest - step
        if not rows:
            raise ValueError(f"No candles returned for {symbol} {interval}")
        ordered = [rows[t] for t in sorted(rows) if t >= start]
        df = pd.DataFrame(ordered).rename(
            columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.set_index("time")[["open", "high", "low", "close", "volume"]].sort_index()
```

### 1b. `config.py` — `BacktestConfig`
Replace its fields with:
```python
@dataclass
class BacktestConfig:
    days: int
    rr: float
    atr_period: int
    atr_mult: float
    warmup_bars: int
```

### 1c. `config.yaml` — `backtest:` block
```yaml
backtest:
  days: 30
  rr: 2.0
  atr_period: 14
  atr_mult: 1.5
  warmup_bars: 215
```

### 1d. Tests
- `tests/test_data_client.py` — add a pagination test. A fake `Info` whose `candle_snapshot(coin, interval, start, end)` returns at most 3 candles per call within `[start,end]` (newest-first slice), so the loop must call repeatedly. Assert the returned DataFrame has all expected unique bars, ascending index, no duplicates. Keep the existing `fetch_candles` tests.
  Example fake:
  ```python
  def test_fetch_candles_days_paginates(monkeypatch):
      import hyperbot.data_client as dc
      step = dc.INTERVAL_MS["15m"]
      # build 7 synthetic candles spaced by `step`, ending "now-ish"
      base = 1_700_000_000_000
      allc = [{"t": base + k*step, "o":"1","h":"2","l":"0.5","c":"1.5","v":"10"} for k in range(7)]
      class FakeInfo:
          def __init__(self,*a,**k): pass
          def candle_snapshot(self, coin, interval, start, end):
              window = [c for c in allc if start <= c["t"] <= end]
              return window[-3:]  # API returns only the most recent <=3 in range
      monkeypatch.setattr(dc, "Info", FakeInfo)
      client = dc.HyperliquidDataClient(testnet=True)
      df = client.fetch_candles_days("BTC", "15m", days=9999)  # large window so start<=base
      assert len(df) == 7
      assert df.index.is_monotonic_increasing
      assert df.index.is_unique
  ```
  (Tune `days` so `start <= base`. The cap-of-3 forces ≥3 paginated calls.)
- `tests/test_config.py` — update assertions to the new `BacktestConfig` (e.g. `cfg.backtest.warmup_bars == 215`, `cfg.backtest.rr == 2.0`, `cfg.backtest.days == 30`, `cfg.backtest.atr_mult == 1.5`). Remove any assertion on removed fields (`metric`, `in_sample_bars`, etc.).

- [ ] Full suite green. Commit: `feat: paginated day-based candle fetch + backtest config`.

---

## STEP 2 — Event-driven backtester (replace `backtest.py`)

Replace `hyperbot/backtest.py` ENTIRELY with the following (removes `expand_grid`/`simulate`/`optimize`/`walk_forward`/deciders):

```python
from __future__ import annotations

import argparse
import json

import pandas as pd
from rich.console import Console
from rich.table import Table

from .config import Config
from .data_client import HyperliquidDataClient
from .strategies import REGISTRY
from .strategies.base import atr
from .strategies.aggregator import aggregate


def run_backtest(df, strategies, *, threshold, min_agree, margin, rr,
                 atr_period=14, atr_mult=1.5, warmup=215):
    """Walk bars one at a time (no lookahead); one trade at a time."""
    atr_series = atr(df, atr_period)
    closes, highs, lows = df["close"].values, df["high"].values, df["low"].values
    index = df.index
    trades = []
    open_trade = None
    n = len(df)
    for i in range(warmup, n):
        if open_trade is not None:
            hi, lo = float(highs[i]), float(lows[i])
            if open_trade["side"] == "long":
                hit_stop, hit_tp = lo <= open_trade["stop"], hi >= open_trade["tp"]
            else:
                hit_stop, hit_tp = hi >= open_trade["stop"], lo <= open_trade["tp"]
            outcome = None
            if hit_stop:           # conservative: stop wins ties
                outcome = "loss"
            elif hit_tp:
                outcome = "win"
            if outcome is not None:
                open_trade.update({
                    "outcome": outcome,
                    "exit_time": str(index[i]),
                    "exit_price": open_trade["stop"] if outcome == "loss" else open_trade["tp"],
                    "bars_held": i - open_trade.pop("_entry_i"),
                    "r_multiple": rr if outcome == "win" else -1.0,
                })
                trades.append(open_trade)
                open_trade = None
            continue  # no new signal while managing/closing a trade

        window = df.iloc[: i + 1]  # only past + current bar -> no lookahead
        sigs = [s.analyze(window) for s in strategies.values()]
        agg = aggregate(sigs, threshold, min_agree, margin)
        if agg.recommendation not in ("long", "short"):
            continue
        a = float(atr_series.iloc[i])
        if pd.isna(a) or a <= 0:
            continue
        entry = float(closes[i])
        stop_dist = atr_mult * a
        if agg.recommendation == "long":
            stop, tp = entry - stop_dist, entry + rr * stop_dist
            agreed = [s.strategy for s in sigs if s.buy_confidence >= threshold]
        else:
            stop, tp = entry + stop_dist, entry - rr * stop_dist
            agreed = [s.strategy for s in sigs if s.sell_confidence >= threshold]
        open_trade = {
            "entry_time": str(index[i]), "side": agg.recommendation,
            "entry": entry, "stop": stop, "tp": tp,
            "outcome": None, "exit_time": None, "exit_price": None,
            "bars_held": 0, "r_multiple": 0.0,
            "strategies_agreed": agreed,
            "confidences": {s.strategy: {"buy": s.buy_confidence, "sell": s.sell_confidence} for s in sigs},
            "_entry_i": i,
        }
    if open_trade is not None:
        open_trade.update({"outcome": "open", "bars_held": n - 1 - open_trade.pop("_entry_i")})
        trades.append(open_trade)
    return trades


def summarize(trades):
    resolved = [t for t in trades if t["outcome"] in ("win", "loss")]
    wins = sum(1 for t in resolved if t["outcome"] == "win")
    total_r = sum(t["r_multiple"] for t in resolved)
    n = len(resolved)
    return {
        "trades": len(trades), "resolved": n, "wins": wins, "losses": n - wins,
        "open": sum(1 for t in trades if t["outcome"] == "open"),
        "win_rate": round(wins / n * 100, 2) if n else 0.0,
        "total_r": round(total_r, 3),
        "expectancy_r": round(total_r / n, 3) if n else 0.0,
    }


def attribution(trades, strategy_names):
    rows = {name: {"agreed_wins": 0, "agreed_losses": 0} for name in strategy_names}
    for t in trades:
        if t["outcome"] not in ("win", "loss"):
            continue
        for name in t["strategies_agreed"]:
            rows.setdefault(name, {"agreed_wins": 0, "agreed_losses": 0})
            rows[name]["agreed_wins" if t["outcome"] == "win" else "agreed_losses"] += 1
    for r in rows.values():
        tot = r["agreed_wins"] + r["agreed_losses"]
        r["win_rate_when_agreed"] = round(r["agreed_wins"] / tot * 100, 2) if tot else 0.0
    return rows


def _print_trades(console, trades):
    table = Table(title="Trades")
    for col in ["#", "entry_time", "side", "entry", "stop", "tp", "outcome", "bars", "R", "#agree"]:
        table.add_column(col, overflow="fold")
    for i, t in enumerate(trades, 1):
        table.add_row(str(i), str(t["entry_time"]), t["side"],
                      f"{t['entry']:.2f}", f"{t['stop']:.2f}", f"{t['tp']:.2f}",
                      t["outcome"] or "open", str(t["bars_held"]),
                      f"{t['r_multiple']:+.1f}", str(len(t["strategies_agreed"])))
    console.print(table)


def _print_attribution(console, attr):
    table = Table(title="Strategy attribution (agreement on resolved trades)")
    for col in ["strategy", "agreed_wins", "agreed_losses", "win% when agreed"]:
        table.add_column(col)
    for name, r in attr.items():
        table.add_row(name, str(r["agreed_wins"]), str(r["agreed_losses"]), f"{r['win_rate_when_agreed']:.1f}")
    console.print(table)


def main():
    cfg = Config.load()
    p = argparse.ArgumentParser(description="Event-driven walk-forward backtester (read-only).")
    p.add_argument("--symbol", default=cfg.symbol)
    p.add_argument("--interval", default=cfg.interval)
    p.add_argument("--days", type=int, default=cfg.backtest.days)
    p.add_argument("--rr", type=float, default=cfg.backtest.rr)
    p.add_argument("--confidence", type=float, default=cfg.aggregator.threshold)
    p.add_argument("--minagree", type=int, default=cfg.aggregator.min_agree)
    p.add_argument("--out", default="backtest_results.json")
    args = p.parse_args()

    console = Console()
    client = HyperliquidDataClient(testnet=cfg.testnet)
    df = client.fetch_candles_days(args.symbol, args.interval, args.days)
    console.print(f"Fetched {len(df)} bars for {args.symbol} {args.interval} (~{args.days}d).")

    strategies = {name: REGISTRY[name](scfg.params) for name, scfg in cfg.strategies.items() if scfg.enabled}
    trades = run_backtest(
        df, strategies,
        threshold=args.confidence, min_agree=args.minagree, margin=cfg.aggregator.margin,
        rr=args.rr, atr_period=cfg.backtest.atr_period, atr_mult=cfg.backtest.atr_mult,
        warmup=cfg.backtest.warmup_bars,
    )
    summary = summarize(trades)
    attr = attribution(trades, list(strategies.keys()))
    _print_trades(console, trades)
    _print_attribution(console, attr)
    console.print(summary)

    result = {
        "symbol": args.symbol, "interval": args.interval, "days": args.days,
        "rr": args.rr, "confidence": args.confidence, "min_agree": args.minagree,
        "margin": cfg.aggregator.margin, "atr_period": cfg.backtest.atr_period,
        "atr_mult": cfg.backtest.atr_mult, "warmup_bars": cfg.backtest.warmup_bars,
        "bars": len(df), "trades": trades, "summary": summary, "attribution": attr,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    console.print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
```

### Tests — rewrite `tests/test_backtest.py` entirely

Use a stub strategy + a synthetic OHLC path; `min_agree=1`, `threshold=50` so one agreeing stub fires. Cover: TP-hit → win (+rr), stop-hit → loss (-1), both-in-bar → loss, one-trade-at-a-time, no-lookahead, open-at-end, summarize, attribution.

```python
import pandas as pd
import pytest

from hyperbot.backtest import run_backtest, summarize, attribution
from hyperbot.strategies.base import Strategy, StrategySignal


def _df(rows):  # rows: list of (o,h,l,c)
    idx = pd.date_range("2021-01-01", periods=len(rows), freq="15min")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": 100.0},
        index=idx,
    )


class _StubLong(Strategy):
    name = "stub"
    def __init__(self, fire_at, seen):
        super().__init__(); self.fire_at = fire_at; self.seen = seen
    def analyze(self, df):
        i = len(df) - 1
        self.seen.append(i)               # record the largest index visible -> no-lookahead check
        if i == self.fire_at:
            return StrategySignal(self.name, 100.0, 0.0, "trending", "fire", df.index[-1])
        return StrategySignal(self.name, 0.0, 0.0, "ranging", "flat", df.index[-1])


def _run(rows, fire_at, warmup=2, rr=2.0):
    seen = []
    trades = run_backtest(_df(rows), {"stub": _StubLong(fire_at, seen)},
                          threshold=50, min_agree=1, margin=15, rr=rr,
                          atr_period=2, atr_mult=1.0, warmup=warmup)
    return trades, seen


def test_long_tp_hit_is_win():
    # flat bars, fire at index 3 (entry=close=100, ATR(2)~? choose path so TP hit next bar)
    rows = [(100,101,99,100)]*3 + [(100,101,99,100)]   # entry bar i=3 close 100
    rows += [(100, 130, 100, 120)]                      # next bar high spikes -> TP
    trades, _ = _run(rows, fire_at=3)
    assert len(trades) == 1
    assert trades[0]["side"] == "long"
    assert trades[0]["outcome"] == "win"
    assert trades[0]["r_multiple"] == 2.0
    assert trades[0]["bars_held"] == 1

def test_long_stop_hit_is_loss():
    rows = [(100,101,99,100)]*3 + [(100,101,99,100)]
    rows += [(100, 101, 70, 80)]                        # next bar low crashes -> stop
    trades, _ = _run(rows, fire_at=3)
    assert trades[0]["outcome"] == "loss"
    assert trades[0]["r_multiple"] == -1.0

def test_both_hit_in_bar_is_loss():
    rows = [(100,101,99,100)]*3 + [(100,101,99,100)]
    rows += [(100, 130, 70, 100)]                       # both TP and stop inside the bar
    trades, _ = _run(rows, fire_at=3)
    assert trades[0]["outcome"] == "loss"

def test_open_at_end_excluded_from_winrate():
    rows = [(100,101,99,100)]*3 + [(100,101,99,100), (100,101,99,100)]
    trades, _ = _run(rows, fire_at=3)
    assert trades[0]["outcome"] == "open"
    assert summarize(trades)["resolved"] == 0

def test_no_lookahead_window_grows_by_one():
    rows = [(100,101,99,100)]*8
    _, seen = _run(rows, fire_at=99)  # never fires
    assert seen == list(range(2, 8))  # warmup=2 .. n-1, strictly one bar at a time

def test_attribution_counts_agreement_on_wins():
    rows = [(100,101,99,100)]*3 + [(100,101,99,100), (100,130,100,120)]
    trades, _ = _run(rows, fire_at=3)
    attr = attribution(trades, ["stub"])
    assert attr["stub"]["agreed_wins"] == 1
    assert attr["stub"]["win_rate_when_agreed"] == 100.0
```
(Implementer: verify the ATR(2) value on the chosen rows actually places stop/TP where the test expects; tune the spike/crash magnitudes until each test passes. ATR here uses `atr_period=2`, `atr_mult=1.0`, so stop_dist = ATR(2) at the entry bar.)

- [ ] Full suite green. Commit: `feat: event-driven walk-forward backtester per PDF p.11`.

---

## STEP 3 — pnl_calc to new schema; retire show_signals; README

### 3a. `pnl_calc.py` — rewrite `compute_stats` for the new JSON
Reads `result["trades"]` (R-based) + `result.get("summary")`. Compute: trades, resolved, wins, losses, open, win_rate, total_r, expectancy_r, avg_bars_held, and best/worst R. Print them. (No equity/Sharpe — those belonged to the old engine.)
```python
def compute_stats(result: dict) -> dict:
    trades = result.get("trades", [])
    resolved = [t for t in trades if t.get("outcome") in ("win", "loss")]
    wins = [t for t in resolved if t["outcome"] == "win"]
    n = len(resolved)
    total_r = sum(t["r_multiple"] for t in resolved)
    held = [t["bars_held"] for t in resolved]
    return {
        "trades": len(trades),
        "resolved": n,
        "wins": len(wins),
        "losses": n - len(wins),
        "open": sum(1 for t in trades if t.get("outcome") == "open"),
        "win_rate": round(len(wins) / n * 100, 2) if n else 0.0,
        "total_r": round(total_r, 3),
        "expectancy_r": round(total_r / n, 3) if n else 0.0,
        "avg_bars_held": round(sum(held) / len(held), 2) if held else 0.0,
        "best_r": max((t["r_multiple"] for t in resolved), default=0.0),
        "worst_r": min((t["r_multiple"] for t in resolved), default=0.0),
    }
```
Update `tests/test_pnl_calc.py` to the new schema (a small `result` dict with win/loss/open trades carrying `outcome`, `r_multiple`, `bars_held`; assert win_rate, total_r, expectancy_r, avg_bars_held).

### 3b. Retire `show_signals`
- Delete `hyperbot/show_signals.py` and `tests/test_show_signals.py` (`git rm`).
- Grep to confirm nothing imports `show_signals`.

### 3c. `README.md`
Update the Usage section: drop `show_signals`; document the new backtest CLI:
```
python -m hyperbot.backtest --symbol BTC --interval 15m --days 30 --rr 2 --confidence 50 --minagree 3
python -m hyperbot.pnl_calc backtest_results.json
python -m hyperbot.analyze
```
Add a one-line note: long backtests (e.g. `--days 180`) recompute strategies per bar and can take several minutes.

- [ ] Full suite green. Commit: `refactor: pnl_calc to R-multiple schema; retire show_signals`.

---

## Self-review checklist
- `--symbol/--interval/--days/--rr/--confidence/--minagree` all present, defaulting from config.
- Pagination collects the full window (no silent truncation); fetch prints bar count.
- No lookahead (strategies see `df.iloc[:i+1]`); one trade at a time; entry at signal-bar close; stop 1.5×ATR(14); TP rr×stop; both-in-bar → loss; warmup 215; open-at-end excluded.
- Records: time, side, entry, stop, tp, outcome, bars_held, strategies_agreed, per-strategy confidences; attribution shows agreement on winners.
- `backtest_results.json` saved; trade table + attribution printed.
- `show_signals` gone; `pnl_calc` reads new schema; full suite green.
```