# Signal Notifier — BTC 1h Push Notifications via ntfy.sh + GitHub Actions

**Date:** 2026-06-10
**Status:** Approved design
**Scope:** Automated hourly signal detection with phone push notifications on state change. No changes to existing engine, backtest, or strategy code.

## Goal

Run the existing BTC 1h 5/5@50 signal check every hour on GitHub's servers (machine-independent), detect state transitions (flat→signal, signal→closed), and push a notification to the user's phone via ntfy.sh. Notify once on entry, once on close — no repeat pings while a trade is open.

## New Files

| File | Responsibility |
|------|----------------|
| `hyperbot/notifier.py` | Signal detection, state diff, ntfy.sh POST, state persistence |
| `state.json` | Last-known signal state committed to repo; updated each run |
| `.github/workflows/signal_notify.yml` | Hourly GitHub Actions cron driver |

No existing files are modified.

## Components

### `hyperbot/notifier.py`

Entry point: `python -m hyperbot.notifier`

**Logic:**
1. Load `state.json` → `prev_state` (`"flat"` or `"open"`) + previous trade dict
2. Fetch BTC 1h candles (Hyperliquid mainnet, 75 days, drop forming bar)
3. Run `run_backtest(...)` with baked-in config (same params as `_signal_check.py`)
4. Determine `curr_state`:
   - `open` if there is an open trade in results
   - `flat` if no open trade AND no new signal on the last closed bar
   - `new_signal` if no open trade AND aggregator fires a valid signal (passes HTF gate)
5. Compare prev → curr and dispatch notification (see table below)
6. Write updated `state.json`

**State transitions that trigger a notification:**

| prev_state | curr_state | Notification |
|---|---|---|
| `flat` | `new_signal` | NEW LONG/SHORT — entry / SL / TP |
| `open` | `flat` | TRADE CLOSED WIN/LOSS — entry → exit / R |
| anything | same | Silent (no notification) |

`new_signal` is a transient state — saved to `state.json` as `"open"` (the trade just opened).

**ntfy.sh call:** HTTP POST to `https://ntfy.sh/{NTFY_TOPIC}` using `requests`. `NTFY_TOPIC` read from environment variable (injected by GitHub Actions from secret).

**Notification payloads:**

| Event | Title | Body | Priority |
|---|---|---|---|
| New LONG | `BTC LONG Signal 🟢` | `entry {price} \| SL {sl} \| TP {tp}` | high |
| New SHORT | `BTC SHORT Signal 🔴` | `entry {price} \| SL {sl} \| TP {tp}` | high |
| Close WIN | `BTC Trade WIN ✅` | `{side} {entry} → {exit} \| +{r}R` | default |
| Close LOSS | `BTC Trade LOSS ❌` | `{side} {entry} → {exit} \| -{r}R` | default |

**Error handling:** ntfy.sh POST failure is logged but does not crash the script (non-fatal). Data fetch failure raises and lets the workflow mark the run as failed (visible in GitHub Actions UI).

### `state.json`

Committed to the repo at initial state `{"state": "flat", "trade": null}`. Updated in-place by `notifier.py` after each run. The GitHub Actions workflow commits it back so state survives across runs.

Schema:
```json
{
  "state": "flat | open",
  "trade": null | {
    "side": "long | short",
    "entry": 63440.0,
    "entry_time": "2026-06-08 18:00:00",
    "stop": 65084.21,
    "tp": 58507.36
  }
}
```

### `.github/workflows/signal_notify.yml`

```
trigger:   cron '2 * * * *'  (2 min past every hour)
           workflow_dispatch  (manual test trigger)
runner:    ubuntu-latest
python:    3.11
steps:
  1. actions/checkout@v4  (with token for push-back)
  2. actions/setup-python@v5
  3. pip install -r requirements.txt
  4. python -m hyperbot.notifier
  5. git add state.json && git commit && git push  (only if state.json changed)
env:
  NTFY_TOPIC: ${{ secrets.NTFY_TOPIC }}
permissions:
  contents: write
```

The commit-back step uses a conditional (`if: state.json is dirty`) so it only pushes when state actually changed, avoiding empty commits on no-signal hours.

## GitHub Secret Required

| Secret | Value |
|---|---|
| `NTFY_TOPIC` | User's private ntfy.sh topic string (set in repo Settings → Secrets → Actions) |

Hyperliquid candle data is public — no API key secret needed.

## Dependencies

`requests` is already in `requirements.txt`. No new dependencies.

## What is NOT in scope

- Notifications for paper trading P&L updates
- ETH or other symbols
- SMS / email / Telegram — ntfy.sh only
- Modifying the existing `_signal_check.py`, `backtest.py`, or any strategy file