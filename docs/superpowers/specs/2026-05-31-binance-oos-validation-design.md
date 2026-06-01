# Binance 1h ETL — Out-of-Sample Validation

**Date:** 2026-05-31
**Status:** Approved design
**Scope:** Add a read-only Binance spot-klines data source to out-of-sample test the existing Hyperliquid-tuned config on a large historical sample. **No live trading. No re-tuning (deferred).**

## Goal & motivation

The current system (4-of-4, FVG v2, 1:3 RR, 75% confidence) was tuned on **Hyperliquid's most-recent ~208 days of 1h data** — a single, in-sample window of only ~55–64 trades/symbol. That's far below the **~400 trades** needed to distinguish a real edge from luck, and it's overfit by construction.

Hyperliquid's candle API is hard-capped at ~5,000 most-recent candles/interval (no older history available), so we pull **Binance spot 1h klines** (free public REST, history back to ~2017) purely as **out-of-sample data the tuning never saw**. Running the locked config across full BTC+ETH history yields **~1,700 pooled trades** — well past the 400-trade bar. The deliverable is: *what does our strategy actually do over a statistically meaningful, unseen sample?* Only after seeing that do we decide whether to tune.

## Decisions (confirmed)

- **Source:** Binance **spot** `BTCUSDT` / `ETHUSDT`, 1h, **max available history** (~2017→now).
- **Purpose:** out-of-sample validation of the **locked** config; accumulate **≥400 pooled trades** before any tuning decision.
- **Re-tuning:** explicitly **out of scope** for this plan (the grid-search engine was removed; re-tuning reintroduces overfitting). Revisited only after seeing OOS results.
- **Boundary:** read-only public REST, no auth, no orders — a deliberate, documented exception to the "Hyperliquid-only data" scope, justified for validation.

## Components

### `hyperbot/binance_data.py` (new, isolated — does not touch `HyperliquidDataClient`)
- `fetch_klines(symbol, interval, start_ms, end_ms) -> list` — Binance public REST `GET /api/v3/klines` (1000 bars/call), paginates forward by advancing `startTime` to the last `closeTime + 1` until reaching `end_ms`/now. No API key. Small sleep between pages for rate limits; dedupe by open time.
- `load_klines(symbol, interval) -> pd.DataFrame` — fetches full history (or reads cache), returns the **same shape as `HyperliquidDataClient.fetch_candles`**: columns `[open, high, low, close, volume]`, datetime index (UTC), ascending. **Caches to `data/binance/<SYMBOL>_<interval>.csv`** (gitignored); reuses cache if present.
- `split_train_test(df, test_fraction=0.3) -> (train_df, test_df)` — chronological split (older = train, recent = test).

Format parity is the key design point: because `load_klines` returns the identical DataFrame contract, it **drops straight into the existing `run_backtest` with zero engine changes**.

### `hyperbot/validate_oos.py` (new CLI)
Loads Binance BTC+ETH 1h, runs the **locked config** (read from `config.yaml`: enabled strategies, `min_agree`, `threshold`, `margin`, `rr`, `atr_*`, `warmup_bars`, `fee`, `slippage`) via `run_backtest`, and reports for each of **full / train / test** segments and **per symbol + pooled**:
- trades, win rate, gross R, cost R, net R, net expectancy
- $100 equity at 1% and 2% risk (compounding), with max drawdown
- pooled trade count (to confirm ≥400)

### Supporting changes
- `requirements.txt`: add `requests`.
- `.gitignore`: add `data/` (cached CSVs are large, not source).

## Data flow
`Binance REST → fetch_klines (paginate) → cache CSV → load_klines → DataFrame → run_backtest (locked config, costs on) → summarize/attribution → validate_oos report (full/train/test, per-symbol + pooled).`

## Error handling
- HTTP/non-200 → raise with status + body snippet; retry transient errors a few times with backoff.
- Empty/short response → stop pagination cleanly.
- Cache read: if the CSV is present and non-empty, use it; otherwise fetch.
- `run_backtest` already guards insufficient warmup (returns no trades for tiny inputs).

## Testing (no network)
- `tests/test_binance_data.py`: monkeypatch `requests.get` to return canned kline pages (forcing ≥2 paginated calls); assert `load_klines` builds the correct `[open,high,low,close,volume]` DataFrame, ascending + deduped; assert pagination advances and terminates. Mirrors `test_data_client.py`'s SDK mock.
- `split_train_test`: assert chronological boundary and fractions on a synthetic frame.
- No live Binance calls in the suite.

## Explicitly out of scope
- Re-optimizing/tuning parameters (deferred until after OOS results).
- Futures/perp klines (spot chosen for max history).
- Any order placement, signing, or authenticated endpoints.
- A standing scheduled/incremental updater (one-shot fetch + cache is enough for validation).