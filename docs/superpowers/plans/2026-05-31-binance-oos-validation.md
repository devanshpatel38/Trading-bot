# Binance 1h OOS Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Binance spot-klines data source and an out-of-sample validation CLI that runs the locked Hyperliquid-tuned config on a ~400-trade slice of unseen BTC+ETH history, while reserving older history as an untouched holdout.

**Architecture:** A new isolated `binance_data.py` (public Binance REST, no auth) returns DataFrames in the **exact same shape** as `HyperliquidDataClient.fetch_candles`, so they drop into the existing `run_backtest` unchanged. A `validate_oos.py` CLI partitions the history (holdout | validation | tuning-overlap) and reports the locked config's stats on the validation slice only.

**Tech Stack:** Python 3.11, `requests`, `pandas`, existing `hyperbot.backtest`/`config`, `pytest`.

---

## Task 1: Scaffolding (dependency + gitignore)

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Add `requests` to `requirements.txt`**

Append a line so the file ends with:
```text
hyperliquid-python-sdk
pandas
numpy
pyyaml
python-dotenv
rich
pytest
requests
```

- [ ] **Step 2: Install it**

Run: `.venv\Scripts\python.exe -m pip install requests`
Expected: installs (or "already satisfied").

- [ ] **Step 3: Add cache dir to `.gitignore`**

Append a line `data/` to `.gitignore` (cached Binance CSVs are large generated data, not source).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: add requests dep and gitignore data/ for binance cache"
```

---

## Task 2: `fetch_klines` + `_rows_to_df` (paginated REST → DataFrame)

**Files:**
- Create: `hyperbot/binance_data.py`
- Test: `tests/test_binance_data.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_binance_data.py`:
```python
import pandas as pd

import hyperbot.binance_data as bd


def _universe(n, start_ms=1_600_000_000_000, step=3_600_000):
    # n synthetic 1h klines: [openTime, open, high, low, close, volume, closeTime, ...]
    out = []
    for i in range(n):
        t = start_ms + i * step
        out.append([t, "10.0", "11.0", "9.0", "10.5", "100.0", t + step - 1, "0", 0, "0", "0", "0"])
    return out


class _FakeResp:
    def __init__(self, data):
        self._data = data
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._data


class _FakeSession:
    """Mimics Binance: returns up to `limit` klines whose openTime is within [startTime, endTime]."""
    def __init__(self, universe):
        self.universe = universe
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        s, e, lim = params["startTime"], params["endTime"], params["limit"]
        rows = [r for r in self.universe if s <= r[0] <= e][:lim]
        return _FakeResp(rows)


def test_fetch_klines_paginates_over_1000_cap():
    uni = _universe(2500)
    fake = _FakeSession(uni)
    rows = bd.fetch_klines("BTCUSDT", "1h", uni[0][0], uni[-1][0] + 1, session=fake)
    assert len(rows) == 2500
    assert fake.calls >= 3            # 1000 + 1000 + 500 => 3 paginated calls
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)  # ascending, no gaps


def test_rows_to_df_shape_types_dedup():
    uni = _universe(5) + _universe(1)  # duplicate first bar appended
    df = bd._rows_to_df(uni)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "time"
    assert str(df["close"].dtype) == "float64"
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique               # duplicate openTime collapsed
    assert len(df) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_binance_data.py -v`
Expected: FAIL — `No module named 'hyperbot.binance_data'`

- [ ] **Step 3: Create `hyperbot/binance_data.py`**

```python
from __future__ import annotations

import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
CACHE_DIR = "data/binance"
INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, session=None) -> list:
    """Binance public klines, paginated forward (1000/call cap). No auth."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval '{interval}'. Choose from {list(INTERVAL_MS)}")
    step = INTERVAL_MS[interval]
    get = (session or requests).get
    rows: list = []
    cursor = start_ms
    while cursor < end_ms:
        resp = get(BASE_URL, params={"symbol": symbol, "interval": interval,
                                     "startTime": cursor, "endTime": end_ms, "limit": 1000}, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance {resp.status_code}: {resp.text[:200]}")
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + step
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < 1000:
            break
        time.sleep(0.25)  # courtesy rate-limit between pages
    return rows


def _rows_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.drop_duplicates("time").set_index("time").sort_index()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_binance_data.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add hyperbot/binance_data.py tests/test_binance_data.py
git commit -m "feat: binance fetch_klines (paginated) + rows_to_df"
```

---

## Task 3: `load_klines` (cache) + `partition` (3-way split)

**Files:**
- Modify: `hyperbot/binance_data.py`
- Test: `tests/test_binance_data.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_binance_data.py`)**

```python
def test_load_klines_uses_cache_without_network(tmp_path, monkeypatch):
    # Pre-seed a cache CSV; load_klines must read it and NOT fetch.
    df = bd._rows_to_df(_universe(10))
    cache = tmp_path / "binance"
    cache.mkdir()
    df.to_csv(cache / "BTCUSDT_1h.csv")

    def _boom(*a, **k):
        raise AssertionError("network must not be called when cache exists")
    monkeypatch.setattr(bd, "fetch_klines", _boom)

    out = bd.load_klines("BTCUSDT", "1h", cache_dir=str(cache))
    assert len(out) == 10
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.name == "time"


def test_partition_three_contiguous_segments():
    idx = pd.date_range("2020-01-01", periods=400 * 24, freq="h")  # 400 days hourly
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    holdout, val, tuning = bd.partition(df, tuning_overlap_days=50, validation_years=200 / 365)
    # tuning = last 50 days, validation = 200 days before that, holdout = remainder (~150 days)
    assert (tuning.index.max() - tuning.index.min()).days == 49
    assert (val.index.max() - val.index.min()).days == 199
    # contiguous + complete: no overlap, full coverage
    assert holdout.index.max() < val.index.min() < tuning.index.min()
    assert len(holdout) + len(val) + len(tuning) == len(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_binance_data.py -k "cache or partition" -v`
Expected: FAIL — `module 'hyperbot.binance_data' has no attribute 'load_klines'` / `partition`

- [ ] **Step 3: Append to `hyperbot/binance_data.py`**

```python
def load_klines(symbol: str, interval: str, cache_dir: str = CACHE_DIR, refresh: bool = False) -> pd.DataFrame:
    """Full-history klines, cached to CSV. Returns the same shape as HyperliquidDataClient.fetch_candles."""
    path = os.path.join(cache_dir, f"{symbol}_{interval}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, parse_dates=["time"], index_col="time")
    start = 1_483_228_800_000          # 2017-01-01 UTC (Binance returns from listing date)
    end = int(time.time() * 1000)
    df = _rows_to_df(fetch_klines(symbol, interval, start, end))
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path)
    return df


def partition(df: pd.DataFrame, tuning_overlap_days: int = 208, validation_years: float = 2.0):
    """Chronological 3-way split: (holdout, validation, tuning)."""
    end = df.index[-1]
    tuning_start = end - pd.Timedelta(days=tuning_overlap_days)
    val_start = tuning_start - pd.Timedelta(days=round(365 * validation_years))
    holdout = df[df.index < val_start]
    validation = df[(df.index >= val_start) & (df.index < tuning_start)]
    tuning = df[df.index >= tuning_start]
    return holdout, validation, tuning
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_binance_data.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add hyperbot/binance_data.py tests/test_binance_data.py
git commit -m "feat: binance load_klines cache + 3-way temporal partition"
```

---

## Task 4: `validate_oos.py` CLI

**Files:**
- Create: `hyperbot/validate_oos.py`
- Test: `tests/test_validate_oos.py`

- [ ] **Step 1: Write the failing test (pure equity helper, no network)**

`tests/test_validate_oos.py`:
```python
from hyperbot.validate_oos import equity_curve


def test_equity_curve_compounds_and_tracks_drawdown():
    # one +2R win then one -1R loss at 10% risk, $100 start
    trades = [{"r_multiple": 2.0}, {"r_multiple": -1.0}]
    final, maxdd = equity_curve(trades, risk=0.10)
    # 100 -> 100*(1+0.2)=120 -> 120*(1-0.1)=108 ; peak 120 -> dd (120-108)/120=10%
    assert round(final, 2) == 108.00
    assert round(maxdd, 2) == 10.00
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_validate_oos.py -v`
Expected: FAIL — `cannot import name 'equity_curve'`

- [ ] **Step 3: Create `hyperbot/validate_oos.py`**

```python
from __future__ import annotations

import argparse

from .config import Config
from .binance_data import load_klines, partition
from .strategies import REGISTRY
from .backtest import run_backtest, summarize


def equity_curve(resolved_trades, risk: float):
    """Compound $100 by net R x risk fraction; return (final_balance, max_drawdown_pct)."""
    bal, peak, maxdd = 100.0, 100.0, 0.0
    for t in resolved_trades:
        bal += t["r_multiple"] * risk * bal
        peak = max(peak, bal)
        maxdd = max(maxdd, (peak - bal) / peak)
    return bal, maxdd * 100.0


def _report(label, res):
    n = len(res)
    if n == 0:
        print(f"  {label}: no resolved trades")
        return n
    wins = sum(1 for t in res if t["outcome"] == "win")
    net = sum(t["r_multiple"] for t in res)
    b1, dd1 = equity_curve(res, 0.01)
    b2, dd2 = equity_curve(res, 0.02)
    print(f"  {label}: trades={n} WR={wins / n * 100:.2f}% netR={net:+.1f} exp={net / n:+.3f} "
          f"| $100@1%=${b1:.2f}({b1 - 100:+.1f}%,DD{dd1:.0f}%) @2%=${b2:.2f}({b2 - 100:+.1f}%,DD{dd2:.0f}%)")
    return n


def main():
    cfg = Config.load()
    A, B = cfg.aggregator, cfg.backtest
    p = argparse.ArgumentParser(description="Out-of-sample validation on Binance spot klines (read-only).")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--tuning-overlap-days", type=int, default=208)
    p.add_argument("--validation-years", type=float, default=2.0)
    args = p.parse_args()

    strats = {n: REGISTRY[n](s.params) for n, s in cfg.strategies.items() if s.enabled}
    print(f"locked config: enabled={list(strats.keys())} min_agree={A.min_agree} "
          f"threshold={A.threshold} rr={B.rr} fee={B.fee} slip={B.slippage}")

    pooled = []
    for sym in [s.strip() for s in args.symbols.split(",")]:
        df = load_klines(sym, args.interval)
        holdout, val, tuning = partition(df, args.tuning_overlap_days, args.validation_years)
        print(f"\n=== {sym} {args.interval} ({len(df)} bars) ===")
        if len(holdout):
            print(f"  RESERVED holdout (NOT analyzed): {holdout.index[0].date()} -> {holdout.index[-1].date()} "
                  f"({len(holdout)} bars)")
        print(f"  validation window: {val.index[0].date()} -> {val.index[-1].date()} ({len(val)} bars)")
        trades = run_backtest(val, strats, threshold=A.threshold, min_agree=A.min_agree, margin=A.margin,
                              rr=B.rr, atr_period=B.atr_period, atr_mult=B.atr_mult,
                              warmup=B.warmup_bars, fee=B.fee, slippage=B.slippage)
        res = [t for t in trades if t["outcome"] in ("win", "loss")]
        pooled.extend(res)
        _report(sym, res)

    print("\n=== POOLED (validation only) ===")
    n = _report("POOLED", pooled)
    print(f"\npooled resolved trades = {n} (target ~400)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_validate_oos.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/validate_oos.py tests/test_validate_oos.py
git commit -m "feat: validate_oos CLI - locked config on Binance validation slice"
```

---

## Task 5: Full-suite + live OOS run

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: ALL pass (previous suite + new binance_data + validate_oos tests).

- [ ] **Step 2: Live OOS validation run (hits Binance public REST)**

Run: `.venv\Scripts\python.exe -m hyperbot.validate_oos --symbols BTCUSDT,ETHUSDT --interval 1h --validation-years 2`
Expected: prints, per symbol, the reserved-holdout range (NOT analyzed) + the validation window + stats, then the POOLED line with `pooled resolved trades` near ~400. First run fetches + caches to `data/binance/`; subsequent runs reuse the cache. If Binance is unreachable, report the error — do not add retries/fallbacks.

- [ ] **Step 3: Confirm cache is gitignored**

Run: `git status --porcelain data/`
Expected: no output (the `data/` cache is ignored).

---

## Self-Review Notes

- **Spec coverage:** `binance_data.py` fetch/load/partition (Tasks 2–3) ✓; `validate_oos.py` analyzing validation slice + printing reserved holdout (Task 4) ✓; `requests` dep + `data/` gitignore (Task 1) ✓; DataFrame format parity with `fetch_candles` (`[open,high,low,close,volume]`, `time` index) ✓; mocked-network tests + partition test (Tasks 2–3) ✓; re-tuning excluded (no optimizer added) ✓.
- **No placeholders:** every step has runnable code/commands.
- **Type consistency:** `fetch_klines(symbol, interval, start_ms, end_ms, session=None) -> list`; `_rows_to_df(rows) -> DataFrame[open,high,low,close,volume]`; `load_klines(symbol, interval, cache_dir, refresh) -> DataFrame`; `partition(df, tuning_overlap_days=208, validation_years=2.0) -> (holdout, validation, tuning)`; `equity_curve(resolved_trades, risk) -> (final, maxdd_pct)`. `run_backtest`/`summarize` used with the same kwargs as elsewhere in the repo.
