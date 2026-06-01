# Binance 1h ETL — Out-of-Sample Validation

**Date:** 2026-05-31
**Status:** Approved design
**Scope:** Add a read-only Binance spot-klines data source to out-of-sample test the existing Hyperliquid-tuned config on a large historical sample. **No live trading. No re-tuning (deferred).**

## Goal & motivation

The current system (4-of-4, FVG v2, 1:3 RR, 75% confidence) was tuned on **Hyperliquid's most-recent ~208 days of 1h data** — a single, in-sample window of only ~55–64 trades/symbol. That's far below the **~400 trades** needed to distinguish a real edge from luck, and it's overfit by construction.

Hyperliquid's candle API is hard-capped at ~5,000 most-recent candles/interval (no older history available), so we pull **Binance spot 1h klines** (free public REST, history back to ~2017) purely as **out-of-sample data the tuning never saw**.

Crucially, we do **not** spend all ~8 years now. We validate on a slice sized to **~400 pooled trades** and **reserve the rest as an untouched holdout** for a final confirmation later — so we never burn our only pristine test set. The deliverable is: *what does our strategy do over a statistically meaningful (~400-trade), unseen sample?* Only after seeing that do we decide whether to tune.

## Decisions (confirmed)

- **Source:** Binance **spot** `BTCUSDT` / `ETHUSDT`, 1h, **max available history** (~2017→now).
- **Purpose:** out-of-sample validation of the **locked** config on a **~400-trade** slice; **reserve the rest as an untouched holdout** for a later final test.
- **Re-tuning:** explicitly **out of scope** for this plan (the grid-search engine was removed; re-tuning reintroduces overfitting). Revisited only after seeing OOS results.

## Data partitioning (temporal)

The full cached history is split into three non-overlapping, chronological segments. Only the **validation** segment is analyzed now.

| Segment | Range (approx) | Use |
|---|---|---|
| **In-sample / tuning** | most-recent ~208 days (HL window) | already used to tune; the overlapping recent Binance months are excluded from validation so nothing leaks |
| **Validation (analyze now)** | the ~2 years immediately before the tuning window (≈ 2023‑11 → 2025‑11) | run the locked config here; target **~400 pooled BTC+ETH trades** |
| **Holdout (reserved, NOT analyzed)** | oldest history (≈ 2017 → 2023‑11) | fetched + cached but never loaded into analysis; pristine final test for later |

Implementation: a `tuning_overlap_days` (default 208) end-offset skips the in-sample overlap, and a `validation_years` (default 2) window defines the validation slice; everything older is the holdout. `validate_oos.py` analyzes only the validation slice and prints the reserved holdout's date range + bar count without touching it.
- **Boundary:** read-only public REST, no auth, no orders — a deliberate, documented exception to the "Hyperliquid-only data" scope, justified for validation.

## Components

### `hyperbot/binance_data.py` (new, isolated — does not touch `HyperliquidDataClient`)
- `fetch_klines(symbol, interval, start_ms, end_ms) -> list` — Binance public REST `GET /api/v3/klines` (1000 bars/call), paginates forward by advancing `startTime` to the last `closeTime + 1` until reaching `end_ms`/now. No API key. Small sleep between pages for rate limits; dedupe by open time.
- `load_klines(symbol, interval) -> pd.DataFrame` — fetches full history (or reads cache), returns the **same shape as `HyperliquidDataClient.fetch_candles`**: columns `[open, high, low, close, volume]`, datetime index (UTC), ascending. **Caches to `data/binance/<SYMBOL>_<interval>.csv`** (gitignored); reuses cache if present.
- `partition(df, tuning_overlap_days=208, validation_years=2) -> (holdout_df, validation_df, tuning_df)` — chronological 3-way split by date (per the partitioning table): `tuning_df` = most-recent `tuning_overlap_days`; `validation_df` = the `validation_years` before that; `holdout_df` = everything older.

Format parity is the key design point: because `load_klines` returns the identical DataFrame contract, it **drops straight into the existing `run_backtest` with zero engine changes**.

### `hyperbot/validate_oos.py` (new CLI)
Loads Binance BTC+ETH 1h, carves out the **validation slice** (per the partition above; default `tuning_overlap_days=208`, `validation_years=2`), runs the **locked config** (read from `config.yaml`: enabled strategies, `min_agree`, `threshold`, `margin`, `rr`, `atr_*`, `warmup_bars`, `fee`, `slippage`) via `run_backtest` on **only that slice**, and reports **per symbol + pooled**:
- trades, win rate, gross R, cost R, net R, net expectancy
- $100 equity at 1% and 2% risk (compounding), with max drawdown
- pooled trade count (to confirm it lands near ~400)

It also prints — but does **not** analyze — the **reserved holdout** range and bar count, so it's explicit that pristine data remains for a final test.

### Supporting changes
- `requirements.txt`: add `requests`.
- `.gitignore`: add `data/` (cached CSVs are large, not source).

## Data flow
`Binance REST → fetch_klines (paginate) → cache CSV → load_klines → DataFrame → partition (holdout | validation | tuning-overlap) → run_backtest on VALIDATION slice only (locked config, costs on) → summarize/attribution → report (per-symbol + pooled); holdout range printed but untouched.`

## Error handling
- HTTP/non-200 → raise with status + body snippet; retry transient errors a few times with backoff.
- Empty/short response → stop pagination cleanly.
- Cache read: if the CSV is present and non-empty, use it; otherwise fetch.
- `run_backtest` already guards insufficient warmup (returns no trades for tiny inputs).

## Testing (no network)
- `tests/test_binance_data.py`: monkeypatch `requests.get` to return canned kline pages (forcing ≥2 paginated calls); assert `load_klines` builds the correct `[open,high,low,close,volume]` DataFrame, ascending + deduped; assert pagination advances and terminates. Mirrors `test_data_client.py`'s SDK mock.
- `partition`: on a synthetic dated frame, assert the three segments are contiguous, non-overlapping, cover the whole frame, and that `validation_df`/`tuning_df` span the expected day-counts (holdout is the remainder).
- No live Binance calls in the suite.

## Explicitly out of scope
- Re-optimizing/tuning parameters (deferred until after OOS results).
- Futures/perp klines (spot chosen for max history).
- Any order placement, signing, or authenticated endpoints.
- A standing scheduled/incremental updater (one-shot fetch + cache is enough for validation).