# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hyperbot` — a **read-only** Hyperliquid **testnet** bot: fetch OHLCV candles, run 5 technical strategies through a signal aggregator, and backtest them walk-forward. **Live trading is deliberately out of scope** — see the read-only boundary below.

## Commands

Windows + PowerShell. A virtualenv lives at `.venv`; always invoke it explicitly so the right interpreter/deps are used:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt   # setup
.venv\Scripts\python.exe -m pytest -q                          # full test suite
.venv\Scripts\python.exe -m pytest tests/test_backtest.py::test_simulate_long_in_uptrend_profits -v   # single test
```

CLIs (all read-only; `analyze`/`backtest` hit the testnet network):

```powershell
.venv\Scripts\python.exe -m hyperbot.analyze                       # one-shot signal table for the latest bar
.venv\Scripts\python.exe -m hyperbot.backtest --out backtest_results.json
.venv\Scripts\python.exe -m hyperbot.pnl_calc backtest_results.json
.venv\Scripts\python.exe -m hyperbot.show_signals backtest_results.json
```

There is no linter or build step configured.

## Architecture (the parts that span files)

**Config-driven.** `config.yaml` + `hyperbot/.env` → `Config.load()` (`config.py`) → typed dataclasses (`Config`, `StrategyConfig`, `AggregatorConfig`, `BacktestConfig`). API keys come only from env, never YAML. Strategy params, per-strategy grids, weights, aggregator thresholds, and backtest windows all flow from here — change behavior in `config.yaml`, not in code.

**Strategy contract.** Every strategy subclasses `Strategy` (`strategies/base.py`) and implements `evaluate(df) -> StrategySignal`. `StrategySignal` carries `buy_confidence`/`sell_confidence` (0–100, clamped in `__post_init__`), `regime`, and a human-readable `reason`. Strategies are pure functions of a candle DataFrame and decide on the **latest bar** (`df.iloc[-1]`). On short input they must return `self.neutral(df, reason)` (a 0/0 signal) — never raise.

**Registry.** `strategies/__init__.py` maps strategy name → class in `REGISTRY`. To add a strategy: create the module, add it to `REGISTRY`, and add a matching block under `strategies:` in `config.yaml`. `backtest.py` and `analyze.py` discover strategies only through `REGISTRY`.

**Indicators** (`strategies/base.py`: `ema`, `rsi`, `atr`, `macd`, `bollinger_bands`) are SMA-based (not Wilder's smoothing) on purpose — it makes them deterministic and hand-checkable in `tests/test_indicators.py`. Keep that property if you edit them.

**Aggregator** (`strategies/aggregator.py`) takes per-strategy signals + weights, computes weighted-mean buy/sell confidence and a weighted-vote `regime`, then applies `buy_threshold`/`sell_threshold` + `margin` to emit a `LONG`/`SHORT`/`FLAT` decision in an `AggregatedSignal`.

**Walk-forward backtest** (`backtest.py`) is the most involved piece:
- `optimize()` grid-searches **each strategy independently** on the in-sample window (linear in total grid points, *not* a joint search over the cross-product).
- `simulate()` then runs the **aggregated** system out-of-sample, carrying equity forward across windows. It steps bar-by-bar, re-running every strategy's `evaluate()` on a growing window — so it is roughly O(n²) per combo. This is why `backtest` is slow on large `lookback`; prefer a smaller `lookback` for smoke tests.
- Output JSON (`trades`, `equity_curve`, `windows`) is the contract consumed by `pnl_calc.py` and `show_signals.py`. The trade-dict keys those two read must stay in sync with what `simulate()` writes.

## Hard constraints

- **Read-only boundary:** `data_client.py` constructs only the SDK `Info` client (testnet base URL, `skip_ws=True`). It must never construct `Exchange`, sign, or place orders. Candle data is public, so empty API keys are fine. Adding any order/signing path violates the project's scope.
- **Git is local-only:** this repo uses `git config --local` and a personal GitHub identity. Never run `git config --global` or otherwise touch global git config. Remote is `origin` (the personal `Trading-bot` repo).

## Test fixtures gotcha

When building synthetic candle DataFrames in tests, construct price `Series` with the **same datetime index** as the DataFrame (`pd.Series(values, index=idx)`). A `Series` with a default RangeIndex dropped into a datetime-indexed `DataFrame` silently realigns to all-NaN, which makes strategies return neutral and masks real test failures.

## Design docs

Specs and implementation plans live under `docs/superpowers/specs/` and `docs/superpowers/plans/` — start there for the rationale behind the current structure.