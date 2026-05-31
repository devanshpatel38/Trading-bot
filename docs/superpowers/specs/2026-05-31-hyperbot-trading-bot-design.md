# Hyperbot — Hyperliquid Testnet Trading Bot (Data + Strategy + Backtest)

**Date:** 2026-05-31
**Status:** Approved design
**Scope:** Data pipeline, strategy scaffold with 5 full strategies, walk-forward backtesting. **No live trading.**

## Goal

A Python trading bot that connects to **Hyperliquid testnet** (read-only), fetches OHLCV candles, runs technical-analysis strategies, aggregates their signals, and backtests them with a walk-forward engine. Live order placement is explicitly **out of scope** for this version — no signing/`Exchange` client is constructed anywhere.

## Decisions (confirmed)

- **Strategies:** all 5 fully implemented (not stubs) plus an aggregator.
- **Hyperliquid access:** official `hyperliquid-python-sdk` (`Info` client, testnet base URL).
- **Backtest:** true walk-forward (rolling in-sample param optimization → out-of-sample test).
- **Default symbol:** `BTC`. **Default interval:** `15m`.
- **Position model (v1):** single symbol, long/short, fixed fractional sizing, fees + slippage from config.

## Project layout

```
Trading-model/
├── hyperbot/
│   ├── __init__.py
│   ├── config.py            # loads config.yaml + .env → typed Config
│   ├── data_client.py       # HyperliquidDataClient (read-only candle fetch)
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py          # StrategySignal, Strategy ABC, indicators
│   │   ├── ema_trend.py
│   │   ├── rsi_meanrev.py
│   │   ├── bb_squeeze.py
│   │   ├── fvg.py
│   │   ├── macd_momentum.py
│   │   └── aggregator.py
│   ├── backtest.py          # walk-forward engine (CLI)
│   ├── analyze.py           # live read-only analysis (CLI)
│   ├── pnl_calc.py          # PnL from backtest JSON (CLI)
│   ├── show_signals.py      # per-trade confidence table (CLI)
│   ├── config.yaml
│   └── .env                 # gitignored (created from .env.example)
├── .env.example             # committed template
├── .gitignore
├── requirements.txt
├── README.md
└── tests/
```

Two files not in the original tree are added because the prompts require them:
`data_client.py` (the read-only data client) and `config.py` (config/env loader).

## Dependencies

Runtime: `hyperliquid-python-sdk`, `pandas`, `numpy`, `pyyaml`, `python-dotenv`, `rich`.
Dev: `pytest`.
The repo is `git init`-ed so `.env` can be gitignored; `.env.example` is the committed template.

## Components

### config.py
Loads `config.yaml` and merges `.env` secrets into a typed `Config` object (dataclasses).
Exposes `Config.load(path)`. Sections: `symbol`, `interval`, `data`, `strategies`
(per-strategy params, param grids, weights, enabled flag), `aggregator` (thresholds, margin),
`backtest` (windows, fee, slippage, sizing, lookback). API keys read from env, never from yaml.

### data_client.py
`HyperliquidDataClient` wraps the SDK `Info` client at the testnet base URL.
- `fetch_candles(symbol, interval, lookback) -> pandas.DataFrame`
  columns `[open, high, low, close, volume]`, timestamp index, ascending.
- Read-only. No `Exchange` client, no signing, no order methods. Candle data is public,
  so empty API keys still work.

### strategies/base.py
- `StrategySignal` dataclass: `buy_confidence: float` (0–100), `sell_confidence: float` (0–100),
  `regime: str`, `reason: str`, plus `strategy: str` and `timestamp` for traceability.
  Confidences are clamped/validated to 0–100.
- `Strategy` ABC: `name: str`, `params: dict`, `evaluate(df) -> StrategySignal`.
- Indicator functions (pure, operate on pandas Series/DataFrame):
  `ema`, `atr`, `macd`, `bollinger_bands`, `rsi`.

### The 5 strategies
Each subclasses `Strategy` and returns a real `StrategySignal`:
- **ema_trend** — EMA trend + pullback entry; regime trending vs ranging.
- **rsi_meanrev** — RSI oversold/overbought mean reversion.
- **bb_squeeze** — Bollinger bandwidth squeeze → breakout anticipation.
- **fvg** — Fair Value Gap (3-candle imbalance, SMC).
- **macd_momentum** — MACD signal cross + histogram momentum.

### strategies/aggregator.py
Combines enabled strategies' signals with per-strategy weights (default equal):
weighted-mean `buy_confidence`/`sell_confidence`, consensus `regime` (weighted vote),
and a decision `LONG` / `SHORT` / `FLAT` from `buy_threshold` / `sell_threshold` and a
minimum buy-vs-sell margin. Returns an aggregated `StrategySignal` plus the decision.

### backtest.py (walk-forward)
Rolling windows over the candle history. For each window:
1. Grid-search each enabled strategy's param grid on the **in-sample** segment, choosing
   params that maximize the metric (default: total return).
2. Apply chosen params to the **out-of-sample** segment; simulate trades via the aggregator.
3. Record out-of-sample trades + chosen params per window.
Out-of-sample trades concatenate into one equity curve. Position model: single symbol,
long/short, fixed fractional sizing, `fee` and `slippage` applied per fill.
Output: JSON `{ trades, equity_curve, windows: [{params, metrics}] }`.

### pnl_calc.py
Reads backtest JSON → total PnL, return %, win rate, max drawdown, Sharpe.

### show_signals.py
Renders a per-trade confidence table (entry/exit, buy/sell confidence, regime, reason)
using `rich`.

### analyze.py
Fetches recent live candles, runs all strategies + aggregator on the latest bar, prints
the current read-only market view. **No order placement.**

## config.yaml (shape)

```yaml
symbol: BTC
interval: 15m
data:
  lookback: 5000
strategies:
  ema_trend:   { enabled: true, weight: 1.0, params: {...}, grid: {...} }
  rsi_meanrev: { enabled: true, weight: 1.0, params: {...}, grid: {...} }
  bb_squeeze:  { enabled: true, weight: 1.0, params: {...}, grid: {...} }
  fvg:         { enabled: true, weight: 1.0, params: {...}, grid: {...} }
  macd_momentum:{ enabled: true, weight: 1.0, params: {...}, grid: {...} }
aggregator:
  buy_threshold: 60
  sell_threshold: 60
  margin: 10
backtest:
  in_sample_bars: 1500
  out_sample_bars: 500
  step: 500
  fee: 0.0005
  slippage: 0.0002
  risk_fraction: 0.1
  metric: total_return
```

## Error handling

- Data client raises a clear error on empty/short candle responses; backtest skips windows
  with insufficient bars and logs them.
- Config load fails fast with a descriptive message on missing/invalid keys.
- Strategies guard against insufficient lookback (return neutral 0/0 signal with a reason).

## Testing

`pytest` unit tests:
- Indicators vs hand-computed reference values (`ema`, `atr`, `macd`, `bollinger_bands`, `rsi`).
- `StrategySignal` validation/clamping.
- Aggregator weighting, regime vote, and threshold decisions.
- A small synthetic-candle smoke test of the walk-forward loop (no network).

## Explicitly out of scope

- Any live order placement, signing, or `Exchange` client construction.
- WebSocket streaming.
- Multi-symbol portfolio backtesting.