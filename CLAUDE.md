# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hyperbot` — a crypto trading system that started as a read-only Hyperliquid **testnet** backtester and has grown a **live execution layer**. Two things now live side by side:

1. **Research/backtest (read-only):** fetch OHLCV candles (Hyperliquid *or* Binance), run 5 technical strategies through an aggregator, and backtest them walk-forward with no lookahead.
2. **Live execution:** `binance_bot.py` runs the **OI-chop strategy** on **Binance USDT-M perpetuals**, placing real (signed) orders — demo endpoint by default, `--mainnet` for real funds.
3. **ML meta-model layer** (`hyperbot/ml/`): a frozen logistic **meta-labeling** model that scores each ungated signal and sizes it by tercile (skip / mid / high) — a research pipeline + a runtime `model_filter` switch on the live bot. Off by default.

The read-only boundary that used to cover the whole project is now a **module-level split**, not a project-wide ban — see Hard constraints.

## Commands

Windows + PowerShell. A virtualenv lives at `.venv`; always invoke it explicitly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt      # live bot + research/backtest deps
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt    # ONLY to train/freeze the ML model (sklearn/lightgbm/xgboost)
.venv\Scripts\python.exe -m pytest -q                          # full test suite
.venv\Scripts\python.exe -m pytest tests/test_regime.py -v     # single test file
.venv\Scripts\python.exe -m pytest tests/test_backtest.py -k tp -v   # single test by name
```

There is no linter or build step configured. `requirements-ml.txt` is **research-only**: the live bot and `reconcile` run the frozen model via pure-numpy inference and need no ML libraries.

**Research / backtest CLIs (read-only):**

```powershell
.venv\Scripts\python.exe -m hyperbot.analyze                   # one-shot signal table for the latest bar (Hyperliquid)
.venv\Scripts\python.exe -m hyperbot.backtest --symbol BTC --interval 15m --days 30 --rr 2 --confidence 50 --minagree 3
.venv\Scripts\python.exe -m hyperbot.backtest_chop             # reproduce the LIVE chop strategy over full history (Binance perp + OI)
.venv\Scripts\python.exe -m hyperbot.reconcile                 # real-price forward-test scorecard since the go-live date
.venv\Scripts\python.exe -m hyperbot.validate_oos --symbols BTCUSDT,ETHUSDT   # OOS validation (holdout/validation split)
.venv\Scripts\python.exe -m hyperbot.pnl_calc backtest_results.json
```

**Live bot (places orders):**

```powershell
.venv\Scripts\python.exe -m hyperbot.binance_bot --dry         # evaluate + print, place NO orders (safe smoke test)
.venv\Scripts\python.exe -m hyperbot.binance_bot               # one idempotent pass on the DEMO endpoint
.venv\Scripts\python.exe -m hyperbot.binance_bot --mainnet     # REAL FUNDS — needs BINANCE_KEY / BINANCE_SECRET
```

**ML meta-model (`hyperbot/ml/`):**

```powershell
.venv\Scripts\python.exe -m hyperbot.ml.status                 # what the frozen model would do on the latest bar (read-only)
.venv\Scripts\python.exe -m hyperbot.ml.freeze --version v2    # (re)train + serialize a frozen artifact (needs requirements-ml.txt)
.venv\Scripts\python.exe -m hyperbot.ml.research               # walk-forward LightGBM/XGBoost/logistic comparison
.venv\Scripts\python.exe -m hyperbot.ml.eth_oos                # cross-asset OOS check (train BTC, test ETH)
```

`backtest`/`backtest_chop` recompute every strategy per bar (no lookahead), so long runs take several minutes (`backtest.py` uses `max_window` to keep the scan O(n)). VPS/cron deployment of the live bot is documented in `deploy.md` (note: Binance geo-blocks US IPs with HTTP 451).

## Architecture (the parts that span files)

**Config-driven.** `config.yaml` + `hyperbot/.env` → `Config.load()` (`config.py`) → typed dataclasses (`Config`, `StrategyConfig`, `AggregatorConfig`, `BacktestConfig`, `OIFilterConfig`, `ModelFilterConfig`). API keys come only from env, never YAML. Strategy params/grids, aggregator settings, backtest settings (incl. `htf_period`, `fee`, `slippage`, `warmup_bars=815` to seed EMA800), and the whole `oi_filter` block all flow from here — change behavior in `config.yaml`, not in code. CLI flags on `backtest` override config values. The `oi_filter.enabled` flag is a **runtime switch**: `true` gates entries to the flat-OI "chop" regime; `false` runs ungated (no OI fetch, take the `chop_min_agree`/5 signal every bar). The `model_filter.enabled` flag is a second runtime switch (see the ML layer below), independent of OI. Flip either on the server — no redeploy.

**Data sources (two).** `data_client.py` = read-only Hyperliquid `Info` client (used by `backtest.py`, `analyze.py`, `notifier.py`). `binance_data.py` = Binance public klines, spot *and* USDT-M perp (`futures=True`), used by the live bot and the chop backtests. `oi_data.py` = Binance open-interest: the **Vision archive** (`data.binance.vision`, >30 days of history, ~1-day publication lag) for the deep 30-day reference, plus the **live endpoint** (`openInterestHist`, ~30-day retention, minutes-fresh) for the trailing window; the live bot merges them (`hybrid_oi`) and **must stand aside if the live OI API is unreachable** (never trade on stale OI).

**Strategy contract.** Every strategy subclasses `Strategy` (`strategies/base.py`) and implements `analyze(df) -> StrategySignal`. Scoring is **four binary 0/25-point components** summed, so `buy_confidence`/`sell_confidence` are always multiples of 25 (0–100) and the `reason` string enumerates each component — keep that traceability when editing. Strategies are pure functions of a candle DataFrame, decide on the **latest bar** (`df.iloc[-1]`), and all indicators are causal (backward-looking). On short input they must return `self.neutral(df, reason)` — never raise.

**Registry.** `strategies/__init__.py` maps name → class in `REGISTRY`. To add a strategy: create the module, add it to `REGISTRY`, and add a matching block under `strategies:` in `config.yaml`. All entry points discover strategies only through `REGISTRY`.

**Indicators** (`strategies/base.py`: `ema`, `rsi`, `atr`, `macd`, `bollinger_bands`) are SMA-based (not Wilder's) on purpose — deterministic and hand-checkable in `tests/test_indicators.py`. Keep that property.

**Aggregator — two functions** (`strategies/aggregator.py`):
- `aggregate()` — the original agreement-based vote: count strategies whose relevant confidence ≥ `threshold`; emit `"long"`/`"short"` when `agree ≥ min_agree` **and** `avg ≥ threshold*0.8` **and** the agreed side beats the other by `margin`, else `"stand_aside"`. Returns an `AggregatedSignal`. No per-strategy weights.
- `aggregate_regime(signals, regime, threshold, chop_min_agree)` — **OI-regime-aware** voting used by the live path. Pure agreement-count gate per regime: `high_fuel`/`profit_taking` need 5/5, `weak_expansion` needs ≥4 incl. EMA or MACD, `chop` needs `chop_min_agree`, `bleeding` needs both mean-reversion strats. Returns `(recommendation, agreed_names)`.

**OI regime classification** (`oi_data.py`): a 30-day **smoothed** OI delta (both endpoints averaged over `avg_hours` to kill reference-spike noise) maps to one of five regimes (`chop` = ±2%). `regime_series()` supports a **hysteresis dead-band** (`chop_hyst_enter`/`exit`, e.g. 1.8/2.2) so a bar teetering on the ±2% edge doesn't flip regime on OI noise — this is what makes the live (provisional OI) and backtest (settled OI) regimes agree near the boundary. `unknown` (no 30-day history) is never tradable.

**Event-driven backtest** (`backtest.py`):
- `run_backtest()` walks bars one at a time from `warmup`, feeding each strategy only the trailing `max_window` bars (or full history) — **no lookahead**. On a signal it enters at the **close of the signal bar**, sets `stop = atr_mult*ATR` and `tp = rr*stop_distance`, and manages **one trade at a time**. A bar touching both stop and TP is a **loss** (conservative). R-multiple accounting; `fee`+`slippage` charged per leg.
- **Regime mode:** pass `regime_series`, `regime_rules` (`REGIME_RULES`), `enabled_regimes`, `chop_min_agree` and it uses `aggregate_regime` + per-regime management (per-regime RR, optional partial scale-out, breakeven move, HTF-filter toggle). The regime is fixed at entry for the trade's life. Without those args it's identical to the plain `aggregate` path.
- **ML meta hook:** an optional `meta` callable — a duck-typed `meta(sigs, agreed, is_long, i) -> (take, tercile, p)` — lets a caller veto an entry (low tercile → stay flat) and tag the trade with a tercile. `backtest.py` stays decoupled (never imports `ml/`); `reconcile.py` injects the frozen model through it so the scorecard reproduces live model-gated trades exactly.
- `data_client.fetch_candles_days()` / `binance_data.fetch_klines()` paginate to avoid silent truncation on long runs.
- `backtest_chop.py` is the canonical **reproducible live backtest** — it imports the *exact* `config.yaml`, `aggregate_regime`, `regime_series`, and strategy classes the live bot uses, so its output is what the bot would have done historically.

**Live bot** (`binance_bot.py`): one **idempotent pass per invocation** (cron-friendly). Plain 1:3 means SL and TP rest on the exchange as `closePosition` orders — nothing to monitor mid-trade. In a position → do nothing; flat → cancel leftover brackets, evaluate the chop signal on the **last closed bar**, and if it fires, market-enter + place SL (`atr_mult`×ATR) and TP (`rr`×). Position sizing risks `max(risk_floor, risk_pct × available_balance)` per trade — **unless `model_filter.enabled`**, in which case `evaluate_signal` scores the fired signal with the frozen model and its tercile drives sizing (low → stand aside; mid/high → `mid`/`high` `_pct`/`_floor`), standing aside on a model load failure. State (`binance_bot_state.json`) tracks `last_entry_bar` and `in_position` — the latter enforces **backtest parity**: no re-entry on the candle a trade just closed on.

**Execution client** (`binance_exec.py`): `BinanceFuturesClient` — the *only* signing/order-placing code. Signed REST for Binance USDT-M perp; demo endpoint (`demo-fapi.binance.com`) by default, mainnet (`fapi.binance.com`) with `--mainnet`. Since 2025-12, STOP/TP go through the Algo Service (`/fapi/v1/algoOrder`). **GETs retry on transient errors; POSTs never retry** (a timed-out order may have filled — retrying could double-fill; the safe failure mode is to raise, log, and stand aside).

**Notifier** (`notifier.py`): runs the backtest engine on fresh candles, detects flat→signal and open→closed transitions, and POSTs to ntfy.sh (`NTFY_TOPIC`). Persists `state.json`. Read-only w.r.t. the exchange.

**Reconcile** (`reconcile.py`): the "bot status" scorecard. Re-runs the exact chop engine over **fresh mainnet candles** (the demo feed's prices drift from real BTC) and scores every trade since the go-live date against real highs/lows. When `model_filter.enabled`, it injects the frozen model through the backtest `meta` hook so the scorecard matches the live model-gated trades (skips low tercile, tags each with `[tercile P=…]`).

**ML meta-model layer** (`hyperbot/ml/`): a **meta-labeling** overlay. The ungated 5-strategy vote is the *primary* signal; a frozen logistic *secondary* model scores each fired signal → `P(win)` → tercile → **sizing** (low = skip, mid = `mid_pct`/`mid_floor`, high = `high_pct`/`high_floor`), not an on/off gate. Modules: `features.py` (causal market-context features), `dataset.py` (builds the labeled ungated dataset; `feature_row` is the **single feature-vector builder shared by training and live** — the parity guarantee), `research.py` (walk-forward LightGBM/XGBoost/logistic), `freeze.py` (trains once and serializes a **frozen JSON artifact** `models/meta_logistic_<version>.json` — plain weights, not a pickle; self-verifies vs the sklearn pipeline to ~1e-16), `model_io.py` (`FrozenModel` = pure-numpy inference + `make_meta_decider` for the backtest hook), `status.py` (read-only "what would the model do now"), plus OOS/equity research scripts. Logistic was chosen over the tree models by a generalization-gap + cross-asset (BTC→ETH→SOL) test. **No OI in the features** (dropped as gate and feature); all features are candle-derived, so live and backtest agree.

`pnl_calc.py` turns backtest output JSON into R-multiple stats. `param_sweep.py` and `ib_breakout.py` are auxiliary research scripts (grid sweeps / an inside-bar breakout experiment).

## Hard constraints

- **Module-level read-only boundary (revised).** Order signing and placement live **only** in `binance_exec.py` (`BinanceFuturesClient`) and are driven **only** by `binance_bot.py`. Everything else — `data_client.py` (Hyperliquid `Info`, `skip_ws=True`), `binance_data.py`, `oi_data.py`, `analyze.py`, `backtest.py`, `notifier.py`, `reconcile.py` — is read-only market data and must **never** sign or place orders. Do not add order/signing paths outside the execution layer.
- **Live-trading safety:** the live bot defaults to the **demo** endpoint; `--mainnet` is real money. Never change that default, never make a POST retry (double-fill risk), and never trade on stale/failed OI (stand aside). Keep `binance_bot.py`'s idempotency and backtest-parity re-entry rule intact.
- **Backtest fidelity:** `backtest_chop.py` must keep importing the live `config.yaml`/`aggregate_regime`/`regime_series` — never fork a parallel copy of the strategy logic, or the "reproducible backtest" guarantee breaks.
- **ML deployment discipline:** the live model is a **frozen, versioned JSON artifact** — never let the live bot (or `reconcile`) retrain on the fly (that reintroduces live/backtest drift). Build features with the shared `ml/dataset.feature_row` in both training and live (never reimplement). The live bot + `reconcile` must stay **ML-library-free** (pure-numpy `FrozenModel`); keep `sklearn`/`lightgbm`/`xgboost` imports confined to the research modules and `requirements-ml.txt`. On a model load failure, stand aside. Retraining is deliberate: `freeze --version vN`, revalidate OOS, then promote.
- **Git is local-only:** this repo uses `git config --local` and a personal GitHub identity. Never run `git config --global`. Remote `origin` is the personal `Trading-bot` repo.

## Test fixtures gotcha

When building synthetic candle DataFrames in tests, construct price `Series` with the **same datetime index** as the DataFrame (`pd.Series(values, index=idx)`). A `Series` with a default RangeIndex dropped into a datetime-indexed `DataFrame` silently realigns to all-NaN, masking real failures.

## Design docs

Specs and implementation plans live under `docs/superpowers/specs/` and `docs/superpowers/plans/` — start there for the rationale behind the current structure.
