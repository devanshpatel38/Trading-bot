# Signal Notifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hourly GitHub Actions–driven BTC signal detection that pushes ntfy.sh phone notifications on state change (flat→signal or signal→closed), with no changes to existing engine files.

**Architecture:** A new `hyperbot/notifier.py` module owns the full notification pipeline: load prior state from `state.json`, run `run_backtest` + aggregator, detect state transitions, POST to ntfy.sh, save new state. A GitHub Actions workflow runs it every hour and commits `state.json` back to the repo so state survives across runs.

**Tech Stack:** Python 3.11, `requests` (already in requirements), GitHub Actions, ntfy.sh free tier.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `state.json` | Create | Persistent signal state committed to repo |
| `hyperbot/notifier.py` | Create | Signal check, state diff, ntfy.sh POST, state I/O |
| `tests/test_notifier.py` | Create | Unit + integration tests for notifier |
| `.github/workflows/signal_notify.yml` | Create | Hourly cron driver, commits state back |

No existing files are modified.

---

## Task 1: Create `state.json`

**Files:**
- Create: `state.json`

There is currently an **open SHORT trade** (entry 63440, entered 2026-06-08 18:00 UTC). Initialize to reflect that so the first run detects a close correctly.

- [ ] **Step 1: Create `state.json`**

```json
{
  "state": "open",
  "trade": {
    "side": "short",
    "entry": 63440.0,
    "entry_time": "2026-06-08 18:00:00",
    "stop": 65084.21,
    "tp": 58507.36
  }
}
```

- [ ] **Step 2: Commit**

```powershell
git add state.json
git commit -m "chore: initial signal state (open short 63440)"
```

---

## Task 2: Unit tests for pure helper functions

**Files:**
- Create: `tests/test_notifier.py`

These tests cover the four pure functions: `load_state`, `save_state`, `send_ntfy`, `get_current_state`. No network or data dependencies.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_notifier.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hyperbot.notifier import load_state, save_state, send_ntfy, get_current_state


# --- load_state ---

def test_load_state_returns_default_when_no_file(tmp_path):
    result = load_state(tmp_path / "missing.json")
    assert result == {"state": "flat", "trade": None}


def test_load_state_reads_existing_file(tmp_path):
    p = tmp_path / "state.json"
    data = {
        "state": "open",
        "trade": {"side": "short", "entry": 63440.0,
                  "entry_time": "2026-06-08 18:00:00",
                  "stop": 65084.21, "tp": 58507.36},
    }
    p.write_text(json.dumps(data))
    result = load_state(p)
    assert result["state"] == "open"
    assert result["trade"]["entry"] == 63440.0


# --- save_state ---

def test_save_state_writes_file(tmp_path):
    p = tmp_path / "state.json"
    state = {"state": "flat", "trade": None}
    save_state(state, p)
    assert json.loads(p.read_text()) == state


def test_save_state_overwrites_existing(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"state": "open", "trade": {}}')
    save_state({"state": "flat", "trade": None}, p)
    assert json.loads(p.read_text())["state"] == "flat"


# --- send_ntfy ---

def test_send_ntfy_posts_to_correct_url():
    with patch("hyperbot.notifier.requests.post") as mock_post:
        send_ntfy("my-topic", "Test Title", "Test body", "high")
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://ntfy.sh/my-topic"
    assert kwargs["headers"]["Title"] == "Test Title"
    assert kwargs["headers"]["Priority"] == "high"
    assert kwargs["data"] == b"Test body"


def test_send_ntfy_non_fatal_on_network_error():
    with patch("hyperbot.notifier.requests.post", side_effect=Exception("timeout")):
        send_ntfy("topic", "title", "body")  # must not raise


# --- get_current_state ---

def test_get_current_state_returns_open_when_open_trade_exists():
    trades = [
        {"outcome": "win", "entry_time": "t1", "side": "long", "entry": 100.0,
         "exit_price": 106.0, "r_multiple": 2.0, "stop": 94.0, "tp": 106.0},
        {"outcome": "open", "entry_time": "t2", "side": "short", "entry": 90.0,
         "stop": 92.0, "tp": 84.0, "bars_held": 3},
    ]
    state, trade, last_resolved = get_current_state(trades)
    assert state == "open"
    assert trade["entry"] == 90.0
    assert last_resolved["entry"] == 100.0


def test_get_current_state_returns_flat_when_all_resolved():
    trades = [
        {"outcome": "win", "entry_time": "t1", "side": "long", "entry": 100.0,
         "exit_price": 106.0, "r_multiple": 2.0, "stop": 94.0, "tp": 106.0},
    ]
    state, trade, last_resolved = get_current_state(trades)
    assert state == "flat"
    assert trade is None
    assert last_resolved["outcome"] == "win"


def test_get_current_state_returns_flat_on_empty_trades():
    state, trade, last_resolved = get_current_state([])
    assert state == "flat"
    assert trade is None
    assert last_resolved is None
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_notifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'hyperbot.notifier'`

- [ ] **Step 3: Create `hyperbot/notifier.py` with only the pure helpers**

```python
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from .config import Config
from .data_client import HyperliquidDataClient
from .strategies import REGISTRY
from .strategies.base import atr, ema
from .strategies.aggregator import aggregate
from .backtest import run_backtest

STATE_PATH = Path("state.json")
NTFY_BASE = "https://ntfy.sh"


def load_state(path: Path = STATE_PATH) -> dict:
    """Returns persisted state dict, or default flat state if file absent."""
    if path.exists():
        return json.loads(path.read_text())
    return {"state": "flat", "trade": None}


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.write_text(json.dumps(state, indent=2))


def send_ntfy(topic: str, title: str, body: str, priority: str = "default") -> None:
    """POST to ntfy.sh. Non-fatal — logs on failure, never raises."""
    try:
        requests.post(
            f"{NTFY_BASE}/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            timeout=10,
        )
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
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_notifier.py -v
```

Expected: all 10 tests PASS.

---

## Task 3: `check_new_signal`, `notify`, and `main`

**Files:**
- Modify: `tests/test_notifier.py` (append new tests)
- Modify: `hyperbot/notifier.py` (append new functions)

- [ ] **Step 1: Append transition tests to `tests/test_notifier.py`**

Add the following to the bottom of `tests/test_notifier.py`:

```python
# --- notify ---

def _open_trade(entry_time="2026-06-08 18:00:00"):
    return {"outcome": "open", "entry_time": entry_time, "side": "short",
            "entry": 63440.0, "stop": 65084.21, "tp": 58507.36, "bars_held": 5}


def _resolved_trade(entry_time="2026-06-08 18:00:00", outcome="win"):
    return {
        "outcome": outcome, "entry_time": entry_time, "side": "short",
        "entry": 63440.0, "stop": 65084.21, "tp": 58507.36,
        "exit_price": 58507.36 if outcome == "win" else 65084.21,
        "r_multiple": 2.96 if outcome == "win" else -1.0,
        "bars_held": 20,
    }


def _new_signal_trade():
    return {"side": "long", "entry": 65000.0, "entry_time": "2026-06-10 10:00:00",
            "stop": 62000.0, "tp": 74000.0}


def test_notify_flat_to_new_signal_sends_high_priority_notification():
    prev = {"state": "flat", "trade": None}
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, [], _new_signal_trade(), topic="test-topic")
    mock_ntfy.assert_called_once()
    title, body, *rest = mock_ntfy.call_args[0]
    assert "LONG" in title
    assert "65000" in body
    assert mock_ntfy.call_args[1].get("priority") == "high" or (
        len(mock_ntfy.call_args[0]) > 3 and mock_ntfy.call_args[0][3] == "high"
    )
    assert new_state["state"] == "open"
    assert new_state["trade"]["entry"] == 65000.0


def test_notify_open_to_flat_sends_win_notification():
    entry_time = "2026-06-08 18:00:00"
    prev = {"state": "open", "trade": {"side": "short", "entry": 63440.0,
                                        "entry_time": entry_time}}
    trades = [_resolved_trade(entry_time=entry_time, outcome="win")]
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, trades, None, topic="test-topic")
    mock_ntfy.assert_called_once()
    title = mock_ntfy.call_args[0][0]
    assert "WIN" in title
    assert new_state["state"] == "flat"


def test_notify_open_to_flat_sends_loss_notification():
    entry_time = "2026-06-08 18:00:00"
    prev = {"state": "open", "trade": {"side": "short", "entry": 63440.0,
                                        "entry_time": entry_time}}
    trades = [_resolved_trade(entry_time=entry_time, outcome="loss")]
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, trades, None, topic="test-topic")
    mock_ntfy.assert_called_once()
    title = mock_ntfy.call_args[0][0]
    assert "LOSS" in title
    assert new_state["state"] == "flat"


def test_notify_open_stays_open_no_notification():
    entry_time = "2026-06-08 18:00:00"
    prev = {"state": "open", "trade": {"side": "short", "entry": 63440.0,
                                        "entry_time": entry_time}}
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, [_open_trade(entry_time)], None, topic="test-topic")
    mock_ntfy.assert_not_called()
    assert new_state["state"] == "open"


def test_notify_flat_stays_flat_no_notification():
    prev = {"state": "flat", "trade": None}
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, [], None, topic="test-topic")
    mock_ntfy.assert_not_called()
    assert new_state["state"] == "flat"


def test_notify_no_topic_updates_state_without_sending():
    prev = {"state": "flat", "trade": None}
    with patch("hyperbot.notifier.send_ntfy") as mock_ntfy:
        new_state = notify(prev, [], _new_signal_trade(), topic=None)
    mock_ntfy.assert_not_called()
    assert new_state["state"] == "open"  # state still advances even in dry-run
```

- [ ] **Step 2: Run new tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_notifier.py -v -k "notify"
```

Expected: `ImportError: cannot import name 'notify' from 'hyperbot.notifier'`

- [ ] **Step 3: Append `check_new_signal`, `notify`, and `main` to `hyperbot/notifier.py`**

Append below the `get_current_state` function:

```python

def check_new_signal(df, strategies: dict, cfg) -> dict | None:
    """Check if the last closed bar fires a fresh signal that passed the HTF gate.
    Returns a trade dict {side, entry, entry_time, stop, tp} or None.
    Called only when run_backtest shows no open trade (avoids double-entry).
    """
    bc = cfg.backtest
    sigs = [s.analyze(df) for s in strategies.values()]
    agg = aggregate(sigs, cfg.aggregator.threshold, cfg.aggregator.min_agree, cfg.aggregator.margin)
    if agg.recommendation not in ("long", "short"):
        return None
    close = float(df["close"].iloc[-1])
    htf_val = float(ema(df["close"], bc.htf_period).iloc[-1])
    atr_val = float(atr(df, bc.atr_period).iloc[-1])
    if agg.recommendation == "long" and close <= htf_val:
        return None
    if agg.recommendation == "short" and close >= htf_val:
        return None
    stop_dist = bc.atr_mult * atr_val
    if agg.recommendation == "long":
        stop = close - stop_dist
        tp = close + bc.rr * stop_dist
    else:
        stop = close + stop_dist
        tp = close - bc.rr * stop_dist
    return {
        "side": agg.recommendation,
        "entry": round(close, 2),
        "entry_time": str(df.index[-1]),
        "stop": round(stop, 2),
        "tp": round(tp, 2),
    }


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
            emoji = "🟢" if t["side"] == "long" else "🔴"
            title = f"BTC {side} Signal {emoji}"
            body = f"entry {t['entry']:.2f} | SL {t['stop']:.2f} | TP {t['tp']:.2f}"
            send_ntfy(topic, title, body, priority="high")
            print(f"[notifier] Sent: {title} — {body}")

        elif prev_state["state"] == "open" and curr_state == "flat":
            prev_entry_time = (prev_state.get("trade") or {}).get("entry_time")
            resolved_match = next(
                (t for t in trades
                 if t["outcome"] in ("win", "loss") and t["entry_time"] == prev_entry_time),
                None,
            )
            if resolved_match:
                outcome = resolved_match["outcome"].upper()
                emoji = "✅" if resolved_match["outcome"] == "win" else "❌"
                title = f"BTC Trade {outcome} {emoji}"
                r = resolved_match["r_multiple"]
                side = resolved_match["side"].upper()
                body = (f"{side} {resolved_match['entry']:.2f} → "
                        f"{resolved_match['exit_price']:.2f} | {r:+.2f}R")
            else:
                prev_trade = prev_state.get("trade") or {}
                title = "BTC Trade Closed"
                side = prev_trade.get("side", "?").upper()
                entry = prev_trade.get("entry", 0.0)
                body = f"{side} {entry:.2f} — outcome unknown"
            send_ntfy(topic, title, body)
            print(f"[notifier] Sent: {title} — {body}")

        else:
            print(f"[notifier] No state change "
                  f"(prev={prev_state['state']}, curr={curr_state}) — silent")

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
    trades = run_backtest(
        df, strategies,
        threshold=cfg.aggregator.threshold,
        min_agree=cfg.aggregator.min_agree,
        margin=cfg.aggregator.margin,
        rr=bc.rr, atr_period=bc.atr_period, atr_mult=bc.atr_mult,
        warmup=bc.warmup_bars, fee=bc.fee, slippage=bc.slippage,
        max_window=600, htf_period=bc.htf_period,
    )

    curr_state_raw, _, _ = get_current_state(trades)
    new_signal_trade = None
    if curr_state_raw == "flat":
        new_signal_trade = check_new_signal(df, strategies, cfg)

    new_state = notify(prev, trades, new_signal_trade, topic)
    save_state(new_state)
    print(f"[notifier] State: {prev['state']} → {new_state['state']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update the import line at the top of `tests/test_notifier.py`**

Change the import from:
```python
from hyperbot.notifier import load_state, save_state, send_ntfy, get_current_state
```
to:
```python
from hyperbot.notifier import load_state, save_state, send_ntfy, get_current_state, notify
```

- [ ] **Step 5: Run the full test suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_notifier.py -v
```

Expected: all 17 tests PASS.

- [ ] **Step 6: Run the full project test suite to verify no regressions**

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected: all existing tests still PASS.

- [ ] **Step 7: Commit**

```powershell
git add hyperbot/notifier.py tests/test_notifier.py
git commit -m "feat: signal notifier (ntfy.sh, state machine, TDD)"
```

---

## Task 4: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/signal_notify.yml`

- [ ] **Step 1: Create the `.github/workflows/` directory and workflow file**

Create `.github/workflows/signal_notify.yml`:

```yaml
name: BTC Signal Notify

on:
  schedule:
    - cron: '2 * * * *'   # 2 min past every hour — 1h candle has closed
  workflow_dispatch:        # manual trigger for testing

permissions:
  contents: write           # needed to commit state.json back

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run signal notifier
        env:
          NTFY_TOPIC: ${{ secrets.NTFY_TOPIC }}
        run: python -m hyperbot.notifier

      - name: Commit state if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git diff --quiet state.json || (
            git add state.json &&
            git commit -m "chore: update signal state [skip ci]" &&
            git push
          )
```

- [ ] **Step 2: Verify YAML is syntactically valid**

```powershell
.venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('.github/workflows/signal_notify.yml').read()); print('YAML OK')"
```

Expected: `YAML OK`

- [ ] **Step 3: Commit**

```powershell
git add .github/workflows/signal_notify.yml
git commit -m "ci: hourly BTC signal notifier workflow (ntfy.sh)"
```

---

## Task 5: Configure GitHub Secret and verify end-to-end

**No code changes — setup and smoke test.**

- [ ] **Step 1: Push all commits to GitHub**

```powershell
git push origin main
```

- [ ] **Step 2: Add `NTFY_TOPIC` secret in GitHub**

1. On your phone, install the **ntfy** app (iOS or Android) from the app store.
2. In the ntfy app, subscribe to a topic — pick a unique name (e.g. `hyperbot-btc-devansh-2026`). The topic name is the only "password" — keep it unguessable.
3. In your browser, go to: `https://github.com/<your-username>/Trading-bot/settings/secrets/actions`
4. Click **New repository secret**
5. Name: `NTFY_TOPIC` — Value: the topic string you chose
6. Click **Add secret**

- [ ] **Step 3: Trigger a manual test run**

1. Go to `https://github.com/<your-username>/Trading-bot/actions`
2. Click **BTC Signal Notify** in the left sidebar
3. Click **Run workflow** → **Run workflow**
4. Watch the run complete (takes ~60s)
5. Check the run logs — you should see `[notifier] State: open → open` (no notification since state hasn't changed)
6. Check that `state.json` was not modified by an extra commit (since state didn't change)

- [ ] **Step 4: Dry-run local smoke test (optional, no NTFY_TOPIC needed)**

```powershell
cd "D:\Personal\Trading-model"
.venv\Scripts\python.exe -m hyperbot.notifier
```

Expected output (approximately):
```
[notifier] NTFY_TOPIC not set — dry run (state updates, no notifications sent)
[notifier] No state change (prev=open, curr=open) — silent
[notifier] State: open → open
```

---

## Self-Review

**Spec coverage check:**
- ✅ `state.json` committed at correct initial state (open SHORT trade)
- ✅ `load_state` / `save_state` implemented and tested
- ✅ `send_ntfy` non-fatal, tested for both success and failure paths
- ✅ `get_current_state` covers open / flat / empty
- ✅ `notify` covers all 5 transitions: flat→signal, open→win, open→loss, open→open (silent), flat→flat (silent)
- ✅ `check_new_signal` applies same HTF gate as `_signal_check.py`
- ✅ `main` drops forming bar (`iloc[:-1]`), uses baked-in config params
- ✅ GitHub Actions: cron `:02`, `workflow_dispatch`, `contents: write`, pip cache, state commit-back with dirty-check
- ✅ `NTFY_TOPIC` from secret, dry-run when unset
- ✅ No existing files modified
- ✅ No new dependencies (requests already in requirements.txt)

**Placeholder scan:** None found — all steps contain complete code.

**Type consistency:** `notify()` receives `(dict, list, dict|None, str|None)` and is imported correctly in tests. `get_current_state` returns `(str, dict|None, dict|None)` — used consistently in both `notify` and `main`.