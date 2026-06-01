# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hyperbot` — a **read-only** Hyperliquid **testnet** bot: fetch OHLCV candles, run 5 technical strategies through a signal aggregator, and backtest them walk-forward. **Live trading is deliberately out of scope** — see the read-only boundary below.

## Commands

Windows + PowerShell. A virtualenv lives at `.venv`; always invoke it explicitly so the right interpreter/deps are used:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt   # setup
.venv\Scripts\python.exe -m pytest -q                          # full test suite
.venv\Scripts\python.exe -m pytest tests/test_backtest.py::test_long_tp_hit_is_win -v   # single test
```

CLIs (all read-only; `analyze`/`backtest` hit the testnet network):

```powershell
.venv\Scripts\python.exe -m hyperbot.analyze                       # one-shot signal table for the latest bar
.venv\Scripts\python.exe -m hyperbot.backtest --symbol BTC --interval 15m --days 30 --rr 2 --confidence 50 --minagree 3
.venv\Scripts\python.exe -m hyperbot.pnl_calc backtest_results.json
```

`backtest` recomputes every strategy per bar (no lookahead), so long runs (e.g. `--days 180` ≈ 17k bars) take several minutes.

There is no linter or build step configured.

## Architecture (the parts that span files)

**Config-driven.** `config.yaml` + `hyperbot/.env` → `Config.load()` (`config.py`) → typed dataclasses (`Config`, `StrategyConfig`, `AggregatorConfig`, `BacktestConfig`). API keys come only from env, never YAML. Strategy params/grids, aggregator settings (`threshold`, `min_agree`, `margin`), and backtest settings (`days`, `rr`, `atr_period`, `atr_mult`, `warmup_bars`) all flow from here — change behavior in `config.yaml`, not in code. CLI flags on `backtest` override the corresponding config values.

**Strategy contract.** Every strategy subclasses `Strategy` (`strategies/base.py`) and implements `analyze(df) -> StrategySignal`. Scoring is **four binary 0/25-point components** summed, so `buy_confidence`/`sell_confidence` are always multiples of 25 (0–100) and the `reason` string enumerates each component (e.g. `htf=25 align=25 pullback=0 candle=25 -> buy 75`) — keep that traceability when editing. Strategies are pure functions of a candle DataFrame and decide on the **latest bar** (`df.iloc[-1]`); all indicators are causal (backward-looking), which is what makes the per-bar backtest correct. On short input they must return `self.neutral(df, reason)` (a 0/0 signal) — never raise.

**Registry.** `strategies/__init__.py` maps strategy name → class in `REGISTRY`. To add a strategy: create the module, add it to `REGISTRY`, and add a matching block under `strategies:` in `config.yaml`. `backtest.py` and `analyze.py` discover strategies only through `REGISTRY`.

**Indicators** (`strategies/base.py`: `ema`, `rsi`, `atr`, `macd`, `bollinger_bands`) are SMA-based (not Wilder's smoothing) on purpose — it makes them deterministic and hand-checkable in `tests/test_indicators.py`. Keep that property if you edit them.

**Aggregator** (`strategies/aggregator.py`) is **agreement-based** (per the design doc): count strategies whose relevant confidence ≥ `threshold`; emit `"long"` when `agree_buy ≥ min_agree` **and** `avg_buy ≥ threshold*0.8` **and** `avg_buy > avg_sell + margin` (mirror for `"short"`), else `"stand_aside"`. Returns an `AggregatedSignal(recommendation, avg_buy, avg_sell, agree_buy, agree_sell, regime, reason, components)`. There are no per-strategy weights.

**Event-driven backtest** (`backtest.py`):
- `run_backtest()` walks bars one at a time from `warmup_bars` (215, so EMA200 is seeded), feeding each strategy only `df.iloc[:i+1]` (**no lookahead**). When the aggregator fires, it enters at the **close of the signal bar**, sets `stop = atr_mult*ATR(atr_period)` and `tp = rr*stop_distance`, then manages **one trade at a time** — no new signals until it closes. A bar that touches both stop and TP counts as a **loss** (conservative). Accounting is in R-multiples (`+rr` win, `-1` loss; still-open trades excluded).
- Per-trade records carry `strategies_agreed` and per-strategy `confidences`; `attribution()` rolls these up to show which strategies agreed on winners.
- `data_client.fetch_candles_days()` paginates backward (~5000/call cap) to avoid silent truncation on long `--days` runs.
- Output JSON (`trades`, `summary`, `attribution`, run params) is consumed by `pnl_calc.py` (R-multiple stats). The backtester prints its own trade + attribution tables.

## Hard constraints

- **Read-only boundary:** `data_client.py` constructs only the SDK `Info` client (testnet base URL, `skip_ws=True`). It must never construct `Exchange`, sign, or place orders. Candle data is public, so empty API keys are fine. Adding any order/signing path violates the project's scope.
- **Git is local-only:** this repo uses `git config --local` and a personal GitHub identity. Never run `git config --global` or otherwise touch global git config. Remote is `origin` (the personal `Trading-bot` repo).

## Test fixtures gotcha

When building synthetic candle DataFrames in tests, construct price `Series` with the **same datetime index** as the DataFrame (`pd.Series(values, index=idx)`). A `Series` with a default RangeIndex dropped into a datetime-indexed `DataFrame` silently realigns to all-NaN, which makes strategies return neutral and masks real test failures.

## Design docs

Specs and implementation plans live under `docs/superpowers/specs/` and `docs/superpowers/plans/` — start there for the rationale behind the current structure.