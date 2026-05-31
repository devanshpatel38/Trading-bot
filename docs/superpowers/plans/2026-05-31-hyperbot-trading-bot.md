# Hyperbot Trading Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Hyperliquid-testnet trading bot that fetches OHLCV candles, runs five technical strategies through a signal aggregator, and backtests them with a walk-forward engine. No live trading.

**Architecture:** A `hyperbot` package with a config/env loader, a read-only SDK-backed data client, a `strategies` subpackage (StrategySignal contract + shared indicators + 5 strategies + aggregator), and four CLI tools (`backtest`, `analyze`, `pnl_calc`, `show_signals`). Strategies are pure functions of a candle DataFrame; the backtester optimizes each strategy's params per-window on in-sample data, then runs the aggregated system out-of-sample.

**Tech Stack:** Python 3.11, `hyperliquid-python-sdk`, `pandas`, `numpy`, `pyyaml`, `python-dotenv`, `rich`, `pytest`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `hyperbot/__init__.py` | Package marker |
| `hyperbot/config.py` | Load `config.yaml` + `.env` into typed `Config` dataclasses |
| `hyperbot/data_client.py` | `HyperliquidDataClient` — read-only candle fetch → DataFrame |
| `hyperbot/strategies/__init__.py` | `REGISTRY` of strategy name → class |
| `hyperbot/strategies/base.py` | `StrategySignal`, `Strategy` ABC, indicators (ema, atr, macd, bollinger_bands, rsi) |
| `hyperbot/strategies/ema_trend.py` | EMA trend pullback strategy |
| `hyperbot/strategies/rsi_meanrev.py` | RSI mean reversion strategy |
| `hyperbot/strategies/bb_squeeze.py` | Bollinger band squeeze strategy |
| `hyperbot/strategies/fvg.py` | Fair Value Gap (SMC) strategy |
| `hyperbot/strategies/macd_momentum.py` | MACD momentum strategy |
| `hyperbot/strategies/aggregator.py` | `AggregatedSignal`, `aggregate()` |
| `hyperbot/backtest.py` | `expand_grid`, `simulate`, `optimize`, `walk_forward`, CLI |
| `hyperbot/pnl_calc.py` | `compute_stats` + CLI |
| `hyperbot/show_signals.py` | `render` per-trade table + CLI |
| `hyperbot/analyze.py` | `analyze()` live read-only view + CLI |
| `hyperbot/config.yaml` | Symbol, interval, strategy params/grids, aggregator, backtest |
| `.env.example` / `.env` | API key template / real keys (gitignored) |
| `tests/` | pytest unit tests |

---

## Task 1: Repo scaffolding

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.env.example`, `README.md`
- Create: `hyperbot/__init__.py`, `hyperbot/strategies/__init__.py` (temporary empty)

- [ ] **Step 1: Create `.gitignore`**

```gitignore
.env
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
backtest_results.json
*.egg-info/
```

- [ ] **Step 2: Create `requirements.txt`**

```text
hyperliquid-python-sdk
pandas
numpy
pyyaml
python-dotenv
rich
pytest
```

- [ ] **Step 3: Create `.env.example`**

```text
HL_API_KEY=
HL_API_SECRET=
```

- [ ] **Step 4: Create package markers**

`hyperbot/__init__.py`:
```python
"""Hyperbot — read-only Hyperliquid testnet trading bot (data + strategies + backtest)."""
```

`hyperbot/strategies/__init__.py` (replaced in Task 11):
```python
```

- [ ] **Step 5: Create a minimal `README.md`**

```markdown
# Hyperbot

Read-only Hyperliquid **testnet** trading bot: fetch candles, run technical strategies, backtest. No live trading.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example hyperbot\.env
```

## Usage
```bash
python -m hyperbot.backtest --out backtest_results.json
python -m hyperbot.pnl_calc backtest_results.json
python -m hyperbot.show_signals backtest_results.json
python -m hyperbot.analyze
```
```

- [ ] **Step 6: Install dependencies and create the venv**

Run: `python -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt`
Expected: installs without error; `hyperliquid`, `pandas`, `rich` available.

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt .env.example README.md hyperbot/__init__.py hyperbot/strategies/__init__.py
git commit -m "chore: scaffold hyperbot project"
```

---

## Task 2: Config loader

**Files:**
- Create: `hyperbot/config.yaml`
- Create: `hyperbot/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Create `hyperbot/config.yaml`**

```yaml
testnet: true
symbol: BTC
interval: 15m
data:
  lookback: 5000
strategies:
  ema_trend:
    enabled: true
    weight: 1.0
    params: {fast: 21, slow: 55, atr_period: 14, pullback_atr: 1.0, trend_min_pct: 0.2}
    grid: {fast: [13, 21], slow: [55, 89]}
  rsi_meanrev:
    enabled: true
    weight: 1.0
    params: {period: 14, oversold: 30, overbought: 70}
    grid: {period: [9, 14], oversold: [25, 30]}
  bb_squeeze:
    enabled: true
    weight: 1.0
    params: {period: 20, num_std: 2.0, squeeze_lookback: 50, squeeze_quantile: 0.25}
    grid: {period: [20, 30]}
  fvg:
    enabled: true
    weight: 1.0
    params: {atr_period: 14, min_gap_atr: 0.25}
    grid: {min_gap_atr: [0.25, 0.5]}
  macd_momentum:
    enabled: true
    weight: 1.0
    params: {fast: 12, slow: 26, signal: 9}
    grid: {fast: [8, 12]}
aggregator:
  buy_threshold: 60
  sell_threshold: 60
  margin: 10
backtest:
  in_sample_bars: 1500
  out_sample_bars: 500
  step: 500
  warmup_bars: 200
  fee: 0.0005
  slippage: 0.0002
  risk_fraction: 0.1
  initial_equity: 10000
  metric: total_return
```

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
from hyperbot.config import Config


def test_config_loads_defaults():
    cfg = Config.load("hyperbot/config.yaml")
    assert cfg.symbol == "BTC"
    assert cfg.interval == "15m"
    assert cfg.lookback == 5000
    assert cfg.testnet is True
    assert set(cfg.strategies.keys()) == {
        "ema_trend", "rsi_meanrev", "bb_squeeze", "fvg", "macd_momentum"
    }
    assert cfg.strategies["ema_trend"].grid["fast"] == [13, 21]
    assert cfg.aggregator.buy_threshold == 60
    assert cfg.backtest.warmup_bars == 200
    assert cfg.backtest.metric == "total_return"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hyperbot.config'`

- [ ] **Step 4: Write `hyperbot/config.py`**

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv


@dataclass
class StrategyConfig:
    enabled: bool
    weight: float
    params: dict
    grid: dict


@dataclass
class AggregatorConfig:
    buy_threshold: float
    sell_threshold: float
    margin: float


@dataclass
class BacktestConfig:
    in_sample_bars: int
    out_sample_bars: int
    step: int
    warmup_bars: int
    fee: float
    slippage: float
    risk_fraction: float
    initial_equity: float
    metric: str


@dataclass
class Config:
    symbol: str
    interval: str
    lookback: int
    testnet: bool
    strategies: dict
    aggregator: AggregatorConfig
    backtest: BacktestConfig
    api_key: str | None = None
    api_secret: str | None = None

    @classmethod
    def load(cls, path: str = "hyperbot/config.yaml", env_path: str = "hyperbot/.env") -> "Config":
        load_dotenv(env_path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        strategies = {
            name: StrategyConfig(
                enabled=bool(s.get("enabled", True)),
                weight=float(s.get("weight", 1.0)),
                params=dict(s.get("params", {})),
                grid=dict(s.get("grid", {})),
            )
            for name, s in raw["strategies"].items()
        }
        return cls(
            symbol=raw["symbol"],
            interval=raw["interval"],
            lookback=int(raw["data"]["lookback"]),
            testnet=bool(raw.get("testnet", True)),
            strategies=strategies,
            aggregator=AggregatorConfig(**raw["aggregator"]),
            backtest=BacktestConfig(**raw["backtest"]),
            api_key=os.getenv("HL_API_KEY"),
            api_secret=os.getenv("HL_API_SECRET"),
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add hyperbot/config.yaml hyperbot/config.py tests/test_config.py
git commit -m "feat: add config.yaml and typed config loader"
```

---

## Task 3: Shared indicators

**Files:**
- Create: `hyperbot/strategies/base.py` (indicators portion)
- Test: `tests/test_indicators.py`

All indicators use SMA-based smoothing where applicable so they are deterministic and hand-checkable.

- [ ] **Step 1: Write the failing test**

`tests/test_indicators.py`:
```python
import pandas as pd
import pytest

from hyperbot.strategies.base import ema, rsi, atr, macd, bollinger_bands


def test_ema_matches_hand_computation():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = ema(s, 3)  # alpha = 0.5, adjust=False
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(1.5)
    assert result.iloc[2] == pytest.approx(2.25)
    assert result.iloc[3] == pytest.approx(3.125)
    assert result.iloc[4] == pytest.approx(4.0625)


def test_rsi_extremes_and_value():
    up = pd.Series(range(1, 30), dtype=float)
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(range(30, 1, -1), dtype=float)
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0)
    mixed = pd.Series([1, 2, 4, 3], dtype=float)
    assert rsi(mixed, 2).iloc[-1] == pytest.approx(66.6667, abs=1e-3)


def test_atr_sma_based():
    df = pd.DataFrame({
        "high": [10, 11, 12],
        "low": [8, 9, 10],
        "close": [9, 10, 11],
    }, dtype=float)
    assert atr(df, 2).iloc[-1] == pytest.approx(2.0)


def test_macd_constant_series_is_zero():
    s = pd.Series([100.0] * 60)
    macd_line, signal_line, hist = macd(s)
    assert macd_line.iloc[-1] == pytest.approx(0.0)
    assert signal_line.iloc[-1] == pytest.approx(0.0)
    assert hist.iloc[-1] == pytest.approx(0.0)


def test_bollinger_bands_known_window():
    s = pd.Series([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
    upper, mid, lower = bollinger_bands(s, period=8, num_std=2)
    assert mid.iloc[-1] == pytest.approx(5.0)
    assert upper.iloc[-1] == pytest.approx(9.0)
    assert lower.iloc[-1] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_indicators.py -v`
Expected: FAIL — cannot import indicators from `base`.

- [ ] **Step 3: Create `hyperbot/strategies/base.py` with indicators**

```python
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_indicators.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add hyperbot/strategies/base.py tests/test_indicators.py
git commit -m "feat: add shared indicators (ema, rsi, atr, macd, bollinger)"
```

---

## Task 4: StrategySignal and Strategy ABC

**Files:**
- Modify: `hyperbot/strategies/base.py` (append signal + ABC)
- Test: `tests/test_base.py`

- [ ] **Step 1: Write the failing test**

`tests/test_base.py`:
```python
import pandas as pd

from hyperbot.strategies.base import StrategySignal, Strategy, last_timestamp


def test_signal_clamps_confidence():
    sig = StrategySignal("x", buy_confidence=150, sell_confidence=-20, regime="trending", reason="r")
    assert sig.buy_confidence == 100.0
    assert sig.sell_confidence == 0.0


def test_last_timestamp():
    df = pd.DataFrame({"close": [1, 2, 3]}, index=pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]))
    assert last_timestamp(df) == pd.Timestamp("2021-01-03")


def test_strategy_merges_default_params():
    class Dummy(Strategy):
        name = "dummy"

        @staticmethod
        def default_params():
            return {"a": 1, "b": 2}

        def evaluate(self, df):
            return self.neutral(df, "noop")

    d = Dummy({"b": 9})
    assert d.params == {"a": 1, "b": 9}
    sig = d.evaluate(pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2021-01-01"])))
    assert sig.buy_confidence == 0.0 and sig.reason == "noop"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_base.py -v`
Expected: FAIL — cannot import `StrategySignal`/`Strategy`/`last_timestamp`.

- [ ] **Step 3: Append to `hyperbot/strategies/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StrategySignal:
    strategy: str
    buy_confidence: float
    sell_confidence: float
    regime: str
    reason: str
    timestamp: object = None

    def __post_init__(self):
        self.buy_confidence = max(0.0, min(100.0, float(self.buy_confidence)))
        self.sell_confidence = max(0.0, min(100.0, float(self.sell_confidence)))


def last_timestamp(df: pd.DataFrame):
    return df.index[-1] if len(df.index) else None


class Strategy(ABC):
    name: str = "base"

    def __init__(self, params: dict | None = None):
        self.params = {**self.default_params(), **(params or {})}

    @staticmethod
    def default_params() -> dict:
        return {}

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        ...

    def neutral(self, df: pd.DataFrame, reason: str) -> StrategySignal:
        return StrategySignal(self.name, 0.0, 0.0, "unknown", reason, last_timestamp(df))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/strategies/base.py tests/test_base.py
git commit -m "feat: add StrategySignal dataclass and Strategy ABC"
```

---

## Task 5: Read-only data client

**Files:**
- Create: `hyperbot/data_client.py`
- Test: `tests/test_data_client.py`

- [ ] **Step 1: Write the failing test (mocked SDK, no network)**

`tests/test_data_client.py`:
```python
import hyperbot.data_client as dc


def test_fetch_candles_builds_dataframe(monkeypatch):
    raw = [
        {"t": 1700000000000, "o": "1.0", "h": "2.0", "l": "0.5", "c": "1.5", "v": "100"},
        {"t": 1700000900000, "o": "1.5", "h": "2.5", "l": "1.0", "c": "2.0", "v": "150"},
    ]

    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def candle_snapshot(self, name, interval, startTime, endTime):
            return raw

    monkeypatch.setattr(dc, "Info", FakeInfo)
    client = dc.HyperliquidDataClient(testnet=True)
    df = client.fetch_candles("BTC", "15m", lookback=2)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[-1] == 2.0
    assert df.index.is_monotonic_increasing


def test_fetch_candles_raises_on_empty(monkeypatch):
    class EmptyInfo:
        def __init__(self, *args, **kwargs):
            pass

        def candle_snapshot(self, *a, **k):
            return []

    monkeypatch.setattr(dc, "Info", EmptyInfo)
    client = dc.HyperliquidDataClient(testnet=True)
    try:
        client.fetch_candles("BTC", "15m", 2)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data_client.py -v`
Expected: FAIL — `No module named 'hyperbot.data_client'`

- [ ] **Step 3: Create `hyperbot/data_client.py`**

```python
from __future__ import annotations

import time

import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class HyperliquidDataClient:
    """Read-only Hyperliquid client. Fetches public candle data; never signs or trades."""

    def __init__(self, testnet: bool = True):
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(base_url=base_url, skip_ws=True)

    def fetch_candles(self, symbol: str, interval: str, lookback: int = 500) -> pd.DataFrame:
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported interval '{interval}'. Choose from {list(INTERVAL_MS)}")
        end = int(time.time() * 1000)
        start = end - lookback * INTERVAL_MS[interval]
        raw = self.info.candle_snapshot(symbol, interval, start, end)
        if not raw:
            raise ValueError(f"No candles returned for {symbol} {interval}")
        df = pd.DataFrame(raw).rename(
            columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.set_index("time")[["open", "high", "low", "close", "volume"]].sort_index()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data_client.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add hyperbot/data_client.py tests/test_data_client.py
git commit -m "feat: add read-only Hyperliquid data client"
```

---

## Task 6: EMA Trend Pullback strategy

**Files:**
- Create: `hyperbot/strategies/ema_trend.py`
- Test: `tests/test_ema_trend.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ema_trend.py`:
```python
import numpy as np
import pandas as pd

from hyperbot.strategies.ema_trend import EmaTrendStrategy


def _ramp_df(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="15min")
    close = pd.Series(values, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_uptrend_pullback_gives_buy():
    values = list(np.linspace(100, 200, 120))
    values[-1] = values[-2] - 1.0  # small pullback on last bar
    sig = EmaTrendStrategy().evaluate(_ramp_df(values))
    assert sig.buy_confidence > sig.sell_confidence
    assert sig.regime == "trending"


def test_insufficient_data_is_neutral():
    sig = EmaTrendStrategy().evaluate(_ramp_df([100, 101, 102]))
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.reason == "insufficient data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ema_trend.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `hyperbot/strategies/ema_trend.py`**

```python
from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, ema, atr, last_timestamp


class EmaTrendStrategy(Strategy):
    name = "ema_trend"

    @staticmethod
    def default_params() -> dict:
        return {"fast": 21, "slow": 55, "atr_period": 14, "pullback_atr": 1.0, "trend_min_pct": 0.2}

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["slow"] + 1:
            return self.neutral(df, "insufficient data")
        close = df["close"]
        fast = ema(close, p["fast"])
        slow = ema(close, p["slow"])
        a = atr(df, p["atr_period"])
        c, f, s, av = float(close.iloc[-1]), float(fast.iloc[-1]), float(slow.iloc[-1]), float(a.iloc[-1])
        if pd.isna(av) or av <= 0:
            return self.neutral(df, "ATR unavailable")
        trend_pct = abs(f - s) / c * 100.0
        regime = "trending" if trend_pct >= p["trend_min_pct"] else "ranging"
        dist = (c - f) / av  # signed distance from fast EMA in ATR units
        buy = sell = 0.0
        if f > s:
            if dist <= p["pullback_atr"]:
                proximity = max(0.0, 1.0 - abs(dist) / p["pullback_atr"])
                buy = min(100.0, 40.0 + 60.0 * proximity)
                reason = f"Uptrend (fast>slow, {trend_pct:.2f}%), pullback {dist:.2f} ATR to fast EMA"
            else:
                buy = 20.0
                reason = f"Uptrend but extended {dist:.2f} ATR above fast EMA"
        elif f < s:
            if dist >= -p["pullback_atr"]:
                proximity = max(0.0, 1.0 - abs(dist) / p["pullback_atr"])
                sell = min(100.0, 40.0 + 60.0 * proximity)
                reason = f"Downtrend (fast<slow, {trend_pct:.2f}%), pullback {dist:.2f} ATR to fast EMA"
            else:
                sell = 20.0
                reason = f"Downtrend but extended {dist:.2f} ATR below fast EMA"
        else:
            reason = "No clear trend"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ema_trend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/strategies/ema_trend.py tests/test_ema_trend.py
git commit -m "feat: add EMA trend pullback strategy"
```

---

## Task 7: RSI Mean Reversion strategy

**Files:**
- Create: `hyperbot/strategies/rsi_meanrev.py`
- Test: `tests/test_rsi_meanrev.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rsi_meanrev.py`:
```python
import pandas as pd

from hyperbot.strategies.rsi_meanrev import RsiMeanRevStrategy


def _df(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="15min")
    close = pd.Series(values, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_falling_price_is_oversold_buy():
    sig = RsiMeanRevStrategy().evaluate(_df(list(range(60, 1, -1))))
    assert sig.buy_confidence == 100.0
    assert sig.sell_confidence == 0.0


def test_rising_price_is_overbought_sell():
    sig = RsiMeanRevStrategy().evaluate(_df(list(range(1, 60))))
    assert sig.sell_confidence == 100.0
    assert sig.buy_confidence == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rsi_meanrev.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `hyperbot/strategies/rsi_meanrev.py`**

```python
from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, rsi, last_timestamp


class RsiMeanRevStrategy(Strategy):
    name = "rsi_meanrev"

    @staticmethod
    def default_params() -> dict:
        return {"period": 14, "oversold": 30.0, "overbought": 70.0}

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["period"] + 1:
            return self.neutral(df, "insufficient data")
        r = float(rsi(df["close"], p["period"]).iloc[-1])
        if pd.isna(r):
            return self.neutral(df, "RSI unavailable")
        buy = sell = 0.0
        regime = "ranging"
        if r <= p["oversold"]:
            buy = min(100.0, 50.0 + (p["oversold"] - r) / max(p["oversold"], 1e-9) * 50.0)
            reason = f"RSI {r:.1f} oversold (<= {p['oversold']})"
        elif r >= p["overbought"]:
            sell = min(100.0, 50.0 + (r - p["overbought"]) / max(100.0 - p["overbought"], 1e-9) * 50.0)
            reason = f"RSI {r:.1f} overbought (>= {p['overbought']})"
        else:
            reason = f"RSI {r:.1f} neutral"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rsi_meanrev.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/strategies/rsi_meanrev.py tests/test_rsi_meanrev.py
git commit -m "feat: add RSI mean reversion strategy"
```

---

## Task 8: Bollinger Band Squeeze strategy

**Files:**
- Create: `hyperbot/strategies/bb_squeeze.py`
- Test: `tests/test_bb_squeeze.py`

- [ ] **Step 1: Write the failing test**

`tests/test_bb_squeeze.py`:
```python
import pandas as pd

from hyperbot.strategies.bb_squeeze import BbSqueezeStrategy


def _df(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="15min")
    close = pd.Series(values, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_flat_series_is_squeeze_upward_bias():
    sig = BbSqueezeStrategy().evaluate(_df([100.0] * 80))
    assert sig.regime == "squeeze"
    assert sig.buy_confidence == 50.0
    assert sig.sell_confidence == 0.0


def test_insufficient_data_is_neutral():
    sig = BbSqueezeStrategy().evaluate(_df([100.0] * 10))
    assert sig.reason == "insufficient data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_bb_squeeze.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `hyperbot/strategies/bb_squeeze.py`**

```python
from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, bollinger_bands, last_timestamp


class BbSqueezeStrategy(Strategy):
    name = "bb_squeeze"

    @staticmethod
    def default_params() -> dict:
        return {"period": 20, "num_std": 2.0, "squeeze_lookback": 50, "squeeze_quantile": 0.25}

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["period"] + p["squeeze_lookback"]:
            return self.neutral(df, "insufficient data")
        close = df["close"]
        upper, mid, lower = bollinger_bands(close, p["period"], p["num_std"])
        bandwidth = (upper - lower) / mid
        cur_bw = float(bandwidth.iloc[-1])
        thresh = float(bandwidth.iloc[-p["squeeze_lookback"]:].quantile(p["squeeze_quantile"]))
        squeezing = cur_bw <= thresh
        c, u, l, m = float(close.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1]), float(mid.iloc[-1])
        buy = sell = 0.0
        if c > u:
            buy, regime = 70.0, "trending"
            reason = f"Breakout above upper band (bw {cur_bw:.4f})"
        elif c < l:
            sell, regime = 70.0, "trending"
            reason = f"Breakout below lower band (bw {cur_bw:.4f})"
        elif squeezing:
            regime = "squeeze"
            if c >= m:
                buy = 50.0
                reason = f"BB squeeze (bw {cur_bw:.4f} <= {thresh:.4f}), upward bias"
            else:
                sell = 50.0
                reason = f"BB squeeze (bw {cur_bw:.4f} <= {thresh:.4f}), downward bias"
        else:
            regime = "ranging"
            reason = f"No squeeze/breakout (bw {cur_bw:.4f})"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_bb_squeeze.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/strategies/bb_squeeze.py tests/test_bb_squeeze.py
git commit -m "feat: add Bollinger band squeeze strategy"
```

---

## Task 9: Fair Value Gap (SMC) strategy

**Files:**
- Create: `hyperbot/strategies/fvg.py`
- Test: `tests/test_fvg.py`

A bullish FVG exists when the current candle's low is above the high of the candle two bars back (`low[-1] > high[-3]`); bearish is the mirror (`high[-1] < low[-3]`).

- [ ] **Step 1: Write the failing test**

`tests/test_fvg.py`:
```python
import pandas as pd

from hyperbot.strategies.fvg import FvgStrategy


def test_bullish_gap_gives_buy():
    n = 20
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    high = [101.0] * n
    low = [99.0] * n
    close = [100.0] * n
    # Create a bullish gap: last candle's low (110) above candle[-3] high (101)
    high[-1], low[-1], close[-1] = 112.0, 110.0, 111.0
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 100.0}, index=idx)
    sig = FvgStrategy().evaluate(df)
    assert sig.buy_confidence > 0.0
    assert sig.sell_confidence == 0.0
    assert sig.regime == "imbalance"


def test_no_gap_is_balanced():
    n = 20
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    close = pd.Series([100.0] * n)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0}, index=idx)
    sig = FvgStrategy().evaluate(df)
    assert sig.buy_confidence == 0.0 and sig.sell_confidence == 0.0
    assert sig.regime == "balanced"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fvg.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `hyperbot/strategies/fvg.py`**

```python
from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, atr, last_timestamp


class FvgStrategy(Strategy):
    name = "fvg"

    @staticmethod
    def default_params() -> dict:
        return {"atr_period": 14, "min_gap_atr": 0.25}

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < max(4, p["atr_period"] + 1):
            return self.neutral(df, "insufficient data")
        high, low = df["high"], df["low"]
        a = float(atr(df, p["atr_period"]).iloc[-1])
        if pd.isna(a) or a <= 0:
            return self.neutral(df, "ATR unavailable")
        high_2, low_2 = float(high.iloc[-3]), float(low.iloc[-3])
        cur_low, cur_high = float(low.iloc[-1]), float(high.iloc[-1])
        buy = sell = 0.0
        regime, reason = "balanced", "No FVG"
        if cur_low > high_2:
            gap_atr = (cur_low - high_2) / a
            if gap_atr >= p["min_gap_atr"]:
                buy = min(100.0, 50.0 + gap_atr * 50.0)
                regime = "imbalance"
                reason = f"Bullish FVG: gap {gap_atr:.2f} ATR above candle[-3] high"
        elif cur_high < low_2:
            gap_atr = (low_2 - cur_high) / a
            if gap_atr >= p["min_gap_atr"]:
                sell = min(100.0, 50.0 + gap_atr * 50.0)
                regime = "imbalance"
                reason = f"Bearish FVG: gap {gap_atr:.2f} ATR below candle[-3] low"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fvg.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/strategies/fvg.py tests/test_fvg.py
git commit -m "feat: add Fair Value Gap (SMC) strategy"
```

---

## Task 10: MACD Momentum strategy

**Files:**
- Create: `hyperbot/strategies/macd_momentum.py`
- Test: `tests/test_macd_momentum.py`

- [ ] **Step 1: Write the failing test**

`tests/test_macd_momentum.py`:
```python
import numpy as np
import pandas as pd

from hyperbot.strategies.macd_momentum import MacdMomentumStrategy


def _df(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="15min")
    close = pd.Series(values, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_rising_series_gives_buy():
    sig = MacdMomentumStrategy().evaluate(_df(list(np.linspace(100, 200, 80))))
    assert sig.buy_confidence > 0.0
    assert sig.sell_confidence == 0.0


def test_insufficient_data_is_neutral():
    sig = MacdMomentumStrategy().evaluate(_df([100.0] * 5))
    assert sig.reason == "insufficient data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_macd_momentum.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `hyperbot/strategies/macd_momentum.py`**

```python
from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategySignal, macd, last_timestamp


class MacdMomentumStrategy(Strategy):
    name = "macd_momentum"

    @staticmethod
    def default_params() -> dict:
        return {"fast": 12, "slow": 26, "signal": 9}

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        p = self.params
        if len(df) < p["slow"] + p["signal"] + 1:
            return self.neutral(df, "insufficient data")
        macd_line, signal_line, hist = macd(df["close"], p["fast"], p["slow"], p["signal"])
        m, s = float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
        h, hp = float(hist.iloc[-1]), float(hist.iloc[-2])
        m_prev, s_prev = float(macd_line.iloc[-2]), float(signal_line.iloc[-2])
        crossed_up = m_prev <= s_prev and m > s
        crossed_dn = m_prev >= s_prev and m < s
        buy = sell = 0.0
        if crossed_up:
            buy, reason = 75.0, "MACD bullish cross"
        elif crossed_dn:
            sell, reason = 75.0, "MACD bearish cross"
        elif m > s and h >= hp:
            buy, reason = 55.0, "MACD above signal, momentum rising"
        elif m < s and h <= hp:
            sell, reason = 55.0, "MACD below signal, momentum falling"
        else:
            reason = "MACD momentum mixed"
        regime = "trending" if max(buy, sell) >= 70.0 else "ranging"
        return StrategySignal(self.name, buy, sell, regime, reason, last_timestamp(df))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_macd_momentum.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/strategies/macd_momentum.py tests/test_macd_momentum.py
git commit -m "feat: add MACD momentum strategy"
```

---

## Task 11: Strategy registry and aggregator

**Files:**
- Modify: `hyperbot/strategies/__init__.py` (add `REGISTRY`)
- Create: `hyperbot/strategies/aggregator.py`
- Test: `tests/test_aggregator.py`

- [ ] **Step 1: Write the failing test**

`tests/test_aggregator.py`:
```python
from hyperbot.strategies import REGISTRY
from hyperbot.strategies.base import StrategySignal
from hyperbot.strategies.aggregator import aggregate, AggregatedSignal


def test_registry_has_all_strategies():
    assert set(REGISTRY.keys()) == {
        "ema_trend", "rsi_meanrev", "bb_squeeze", "fvg", "macd_momentum"
    }


def test_aggregate_weighted_mean_and_long_decision():
    sigs = [
        StrategySignal("a", 80, 0, "trending", "r1"),
        StrategySignal("b", 40, 0, "trending", "r2"),
    ]
    agg = aggregate(sigs, {"a": 1.0, "b": 1.0}, buy_threshold=60, sell_threshold=60, margin=10)
    assert isinstance(agg, AggregatedSignal)
    assert agg.buy_confidence == 60.0
    assert agg.sell_confidence == 0.0
    assert agg.regime == "trending"
    assert agg.decision == "LONG"


def test_aggregate_flat_when_below_threshold():
    sigs = [StrategySignal("a", 50, 0, "ranging", "r")]
    agg = aggregate(sigs, {"a": 1.0}, buy_threshold=60, sell_threshold=60, margin=10)
    assert agg.decision == "FLAT"


def test_aggregate_short_decision():
    sigs = [StrategySignal("a", 0, 90, "trending", "r")]
    agg = aggregate(sigs, {"a": 1.0}, buy_threshold=60, sell_threshold=60, margin=10)
    assert agg.decision == "SHORT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_aggregator.py -v`
Expected: FAIL — `REGISTRY` / `aggregate` not importable.

- [ ] **Step 3: Replace `hyperbot/strategies/__init__.py`**

```python
from .base import Strategy, StrategySignal, last_timestamp
from .ema_trend import EmaTrendStrategy
from .rsi_meanrev import RsiMeanRevStrategy
from .bb_squeeze import BbSqueezeStrategy
from .fvg import FvgStrategy
from .macd_momentum import MacdMomentumStrategy

REGISTRY = {
    "ema_trend": EmaTrendStrategy,
    "rsi_meanrev": RsiMeanRevStrategy,
    "bb_squeeze": BbSqueezeStrategy,
    "fvg": FvgStrategy,
    "macd_momentum": MacdMomentumStrategy,
}

__all__ = ["Strategy", "StrategySignal", "last_timestamp", "REGISTRY"]
```

- [ ] **Step 4: Create `hyperbot/strategies/aggregator.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AggregatedSignal:
    buy_confidence: float
    sell_confidence: float
    regime: str
    decision: str  # "LONG" | "SHORT" | "FLAT"
    reason: str
    components: list = field(default_factory=list)


def aggregate(signals, weights, buy_threshold, sell_threshold, margin) -> AggregatedSignal:
    total_w = sum(max(0.0, weights.get(s.strategy, 1.0)) for s in signals)
    if not signals or total_w <= 0:
        return AggregatedSignal(0.0, 0.0, "unknown", "FLAT", "no active strategies", [])
    buy = sum(s.buy_confidence * weights.get(s.strategy, 1.0) for s in signals) / total_w
    sell = sum(s.sell_confidence * weights.get(s.strategy, 1.0) for s in signals) / total_w
    regime_scores: dict[str, float] = {}
    for s in signals:
        regime_scores[s.regime] = regime_scores.get(s.regime, 0.0) + weights.get(s.strategy, 1.0)
    regime = max(regime_scores.items(), key=lambda kv: kv[1])[0]
    if buy >= buy_threshold and (buy - sell) >= margin:
        decision = "LONG"
    elif sell >= sell_threshold and (sell - buy) >= margin:
        decision = "SHORT"
    else:
        decision = "FLAT"
    reason = f"buy={buy:.1f} sell={sell:.1f} regime={regime} -> {decision}"
    return AggregatedSignal(round(buy, 4), round(sell, 4), regime, decision, reason, list(signals))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_aggregator.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 6: Commit**

```bash
git add hyperbot/strategies/__init__.py hyperbot/strategies/aggregator.py tests/test_aggregator.py
git commit -m "feat: add strategy registry and signal aggregator"
```

---

## Task 12: Walk-forward backtest engine

**Files:**
- Create: `hyperbot/backtest.py`
- Test: `tests/test_backtest.py`

The engine optimizes each strategy's params independently on each in-sample window (linear in total grid points), then runs the aggregated system out-of-sample, carrying equity forward across windows.

- [ ] **Step 1: Write the failing test**

`tests/test_backtest.py`:
```python
import numpy as np
import pandas as pd

from hyperbot.config import (
    Config, StrategyConfig, AggregatorConfig, BacktestConfig
)
from hyperbot.backtest import expand_grid, simulate, metric_value, walk_forward
from hyperbot.strategies.base import Strategy, StrategySignal


def _ramp_df(n):
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    close = pd.Series(np.linspace(100, 300, n))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


class _AlwaysLong(Strategy):
    name = "always_long"

    def evaluate(self, df):
        return StrategySignal(self.name, 100.0, 0.0, "trending", "stub", df.index[-1])


def test_expand_grid_cartesian_product():
    combos = expand_grid({"a": 1, "b": 2}, {"a": [1, 2], "b": [3]})
    assert len(combos) == 2
    assert {"a": 1, "b": 3} in combos
    assert {"a": 2, "b": 3} in combos


def test_expand_grid_empty_returns_base():
    assert expand_grid({"a": 1}, {}) == [{"a": 1}]


def test_simulate_long_in_uptrend_profits():
    df = _ramp_df(50)
    agg_cfg = {"buy_threshold": 60, "sell_threshold": 60, "margin": 10}
    bt_cfg = {"fee": 0.0, "slippage": 0.0, "risk_fraction": 1.0}
    trades, curve, equity = simulate(
        df, {"always_long": _AlwaysLong()}, {"always_long": 1.0},
        agg_cfg, bt_cfg, start_offset=5, initial_equity=10000.0,
    )
    assert equity > 10000.0
    assert len(trades) >= 1
    assert trades[0]["side"] == "LONG"


def test_metric_value_total_return():
    trades = [{"pnl": 500.0, "return_pct": 5.0}, {"pnl": -100.0, "return_pct": -1.0}]
    assert metric_value(trades, 10000.0, "total_return") == 0.04


def test_walk_forward_smoke():
    df = _ramp_df(100)
    cfg = Config(
        symbol="BTC", interval="15m", lookback=100, testnet=True,
        strategies={
            "ema_trend": StrategyConfig(
                enabled=True, weight=1.0,
                params={"fast": 5, "slow": 10, "atr_period": 5, "pullback_atr": 2.0, "trend_min_pct": 0.0},
                grid={"fast": [5, 8]},
            )
        },
        aggregator=AggregatorConfig(buy_threshold=40, sell_threshold=40, margin=5),
        backtest=BacktestConfig(
            in_sample_bars=40, out_sample_bars=20, step=20, warmup_bars=15,
            fee=0.0005, slippage=0.0002, risk_fraction=0.1,
            initial_equity=10000.0, metric="total_return",
        ),
    )
    result = walk_forward(df, cfg)
    assert "trades" in result and "equity_curve" in result and "windows" in result
    assert len(result["windows"]) >= 1
    assert result["windows"][0]["params"]["ema_trend"]["fast"] in (5, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v`
Expected: FAIL — `No module named 'hyperbot.backtest'`

- [ ] **Step 3: Create `hyperbot/backtest.py`**

```python
from __future__ import annotations

import argparse
import itertools
import json
import math

import pandas as pd

from .config import Config
from .data_client import HyperliquidDataClient
from .strategies import REGISTRY
from .strategies.aggregator import aggregate


def make_strategy(name: str, params: dict):
    return REGISTRY[name](params)


def expand_grid(base: dict, grid: dict) -> list[dict]:
    if not grid:
        return [dict(base)]
    keys = list(grid.keys())
    out = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combo = dict(base)
        combo.update(dict(zip(keys, values)))
        out.append(combo)
    return out


def simulate(df, strategies, weights, agg_cfg, bt_cfg, start_offset, initial_equity):
    equity = initial_equity
    position = 0  # -1 short, 0 flat, 1 long
    size = 0.0
    open_trade = None
    trades, equity_curve = [], []
    closes = df["close"].values
    index = df.index

    def close_position(exit_price, exit_time, exit_reason):
        nonlocal equity, position, size, open_trade
        gross = position * (exit_price - open_trade["entry_price"]) * size
        fee = bt_cfg["fee"] * abs(size) * (open_trade["entry_price"] + exit_price)
        pnl = gross - fee
        equity += pnl
        notional = open_trade["entry_price"] * size
        open_trade.update({
            "exit_time": str(exit_time),
            "exit_price": exit_price,
            "pnl": pnl,
            "return_pct": (pnl / notional * 100.0) if notional else 0.0,
            "exit_reason": exit_reason,
            "equity_after": equity,
        })
        trades.append(open_trade)
        position, size, open_trade = 0, 0.0, None

    for i in range(start_offset, len(df)):
        window = df.iloc[: i + 1]
        sigs = [s.evaluate(window) for s in strategies.values()]
        agg = aggregate(sigs, weights, agg_cfg["buy_threshold"], agg_cfg["sell_threshold"], agg_cfg["margin"])
        price = float(closes[i])
        t = index[i]
        target = 1 if agg.decision == "LONG" else (-1 if agg.decision == "SHORT" else 0)
        if target != position:
            if position != 0:
                exit_price = price * (1 - position * bt_cfg["slippage"])
                close_position(exit_price, t, agg.reason)
            if target != 0:
                entry_price = price * (1 + target * bt_cfg["slippage"])
                size = (bt_cfg["risk_fraction"] * equity) / entry_price
                position = target
                open_trade = {
                    "side": "LONG" if target == 1 else "SHORT",
                    "entry_time": str(t),
                    "entry_price": entry_price,
                    "size": size,
                    "buy_confidence": agg.buy_confidence,
                    "sell_confidence": agg.sell_confidence,
                    "regime": agg.regime,
                    "entry_reason": agg.reason,
                }
        equity_curve.append({"time": str(t), "equity": equity})

    if position != 0:
        price = float(closes[-1])
        exit_price = price * (1 - position * bt_cfg["slippage"])
        close_position(exit_price, index[-1], "end of segment")
        equity_curve.append({"time": str(index[-1]), "equity": equity})

    return trades, equity_curve, equity


def metric_value(trades, initial_equity, metric) -> float:
    if not trades:
        return 0.0
    total_pnl = sum(t["pnl"] for t in trades)
    if metric == "sharpe":
        rets = [t["return_pct"] / 100.0 for t in trades]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var)
        return mean / std * math.sqrt(len(rets)) if std > 0 else 0.0
    return total_pnl / initial_equity


def _agg_cfg(cfg: Config) -> dict:
    return {
        "buy_threshold": cfg.aggregator.buy_threshold,
        "sell_threshold": cfg.aggregator.sell_threshold,
        "margin": cfg.aggregator.margin,
    }


def _bt_cfg(cfg: Config) -> dict:
    return {
        "fee": cfg.backtest.fee,
        "slippage": cfg.backtest.slippage,
        "risk_fraction": cfg.backtest.risk_fraction,
    }


def optimize(is_df, cfg: Config) -> dict:
    best = {}
    for name, scfg in cfg.strategies.items():
        if not scfg.enabled:
            continue
        combos = expand_grid(scfg.params, scfg.grid)
        best_metric, best_params = float("-inf"), combos[0]
        for combo in combos:
            strat = make_strategy(name, combo)
            trades, _, _ = simulate(
                is_df, {name: strat}, {name: 1.0}, _agg_cfg(cfg), _bt_cfg(cfg),
                cfg.backtest.warmup_bars, cfg.backtest.initial_equity,
            )
            m = metric_value(trades, cfg.backtest.initial_equity, cfg.backtest.metric)
            if m > best_metric:
                best_metric, best_params = m, combo
        best[name] = best_params
    return best


def walk_forward(df, cfg: Config) -> dict:
    bt = cfg.backtest
    n = len(df)
    weights = {name: s.weight for name, s in cfg.strategies.items() if s.enabled}
    all_trades, full_curve, windows = [], [], []
    equity = bt.initial_equity
    i = widx = 0
    while i + bt.in_sample_bars + bt.out_sample_bars <= n:
        is_df = df.iloc[i : i + bt.in_sample_bars]
        best = optimize(is_df, cfg)
        oos_start = i + bt.in_sample_bars
        seg_start = max(0, oos_start - bt.warmup_bars)
        seg = df.iloc[seg_start : oos_start + bt.out_sample_bars]
        offset = oos_start - seg_start
        strategies = {name: make_strategy(name, best[name]) for name in best}
        trades, curve, equity = simulate(
            seg, strategies, weights, _agg_cfg(cfg), _bt_cfg(cfg), offset, equity
        )
        all_trades.extend(trades)
        full_curve.extend(curve)
        windows.append({
            "index": widx,
            "in_sample_start": int(i),
            "oos_start": int(oos_start),
            "params": best,
            "trades": len(trades),
            "end_equity": equity,
        })
        i += bt.step
        widx += 1
    return {
        "trades": all_trades,
        "equity_curve": full_curve,
        "windows": windows,
        "initial_equity": bt.initial_equity,
        "final_equity": equity,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest (read-only).")
    parser.add_argument("--config", default="hyperbot/config.yaml")
    parser.add_argument("--out", default="backtest_results.json")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    client = HyperliquidDataClient(testnet=cfg.testnet)
    df = client.fetch_candles(cfg.symbol, cfg.interval, cfg.lookback)
    result = walk_forward(df, cfg)
    result["symbol"] = cfg.symbol
    result["interval"] = cfg.interval
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"Wrote {len(result['trades'])} trades, final equity {result['final_equity']:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add hyperbot/backtest.py tests/test_backtest.py
git commit -m "feat: add walk-forward backtest engine and CLI"
```

---

## Task 13: PnL calculator

**Files:**
- Create: `hyperbot/pnl_calc.py`
- Test: `tests/test_pnl_calc.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pnl_calc.py`:
```python
from hyperbot.pnl_calc import compute_stats


def test_compute_stats_basic():
    result = {
        "initial_equity": 10000.0,
        "final_equity": 10400.0,
        "trades": [
            {"pnl": 500.0, "return_pct": 5.0},
            {"pnl": -100.0, "return_pct": -1.0},
        ],
        "equity_curve": [
            {"time": "t0", "equity": 10000.0},
            {"time": "t1", "equity": 10500.0},
            {"time": "t2", "equity": 10400.0},
        ],
    }
    stats = compute_stats(result)
    assert stats["trades"] == 2
    assert stats["total_pnl"] == 400.0
    assert stats["win_rate"] == 50.0
    assert stats["return_pct"] == 4.0
    assert stats["max_drawdown_pct"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pnl_calc.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `hyperbot/pnl_calc.py`**

```python
from __future__ import annotations

import argparse
import json
import math


def compute_stats(result: dict) -> dict:
    trades = result.get("trades", [])
    initial = float(result.get("initial_equity", 0.0))
    final = float(result.get("final_equity", initial))
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / len(trades) * 100.0 if trades else 0.0

    curve = [p["equity"] for p in result.get("equity_curve", [])] or [initial, final]
    peak, max_dd = curve[0], 0.0
    for equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)

    rets = [t["return_pct"] / 100.0 for t in trades]
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var)
        sharpe = mean / std * math.sqrt(len(rets)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round((final - initial) / initial * 100.0, 2) if initial else 0.0,
        "win_rate": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "final_equity": round(final, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute PnL stats from backtest JSON.")
    parser.add_argument("results", help="Path to backtest results JSON")
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as fh:
        result = json.load(fh)
    stats = compute_stats(result)
    for key, value in stats.items():
        print(f"{key:>18}: {value}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pnl_calc.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/pnl_calc.py tests/test_pnl_calc.py
git commit -m "feat: add PnL calculator"
```

---

## Task 14: Per-trade signals table

**Files:**
- Create: `hyperbot/show_signals.py`
- Test: `tests/test_show_signals.py`

- [ ] **Step 1: Write the failing test**

`tests/test_show_signals.py`:
```python
from rich.table import Table

from hyperbot.show_signals import render


def test_render_returns_table_with_rows():
    result = {
        "trades": [
            {
                "side": "LONG", "entry_time": "t0", "entry_price": 100.0, "exit_price": 110.0,
                "pnl": 10.0, "buy_confidence": 80.0, "sell_confidence": 0.0,
                "regime": "trending", "entry_reason": "x",
            }
        ]
    }
    table = render(result)
    assert isinstance(table, Table)
    assert table.row_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_show_signals.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `hyperbot/show_signals.py`**

```python
from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

COLUMNS = ["#", "side", "entry_time", "entry", "exit", "pnl", "buy", "sell", "regime", "reason"]


def render(result: dict) -> Table:
    table = Table(title="Per-trade confidence")
    for col in COLUMNS:
        table.add_column(col, overflow="fold")
    for i, t in enumerate(result.get("trades", []), 1):
        table.add_row(
            str(i),
            t["side"],
            str(t["entry_time"]),
            f"{t['entry_price']:.2f}",
            f"{t.get('exit_price', float('nan')):.2f}",
            f"{t.get('pnl', 0.0):.2f}",
            f"{t['buy_confidence']:.0f}",
            f"{t['sell_confidence']:.0f}",
            t["regime"],
            t["entry_reason"],
        )
    return table


def main():
    parser = argparse.ArgumentParser(description="Render per-trade confidence table.")
    parser.add_argument("results", help="Path to backtest results JSON")
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as fh:
        result = json.load(fh)
    Console().print(render(result))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_show_signals.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/show_signals.py tests/test_show_signals.py
git commit -m "feat: add per-trade signals table"
```

---

## Task 15: Live read-only analysis

**Files:**
- Create: `hyperbot/analyze.py`
- Test: `tests/test_analyze.py`

- [ ] **Step 1: Write the failing test (no network — calls `analyze()` directly)**

`tests/test_analyze.py`:
```python
import numpy as np
import pandas as pd

from hyperbot.config import Config, StrategyConfig, AggregatorConfig, BacktestConfig
from hyperbot.analyze import analyze
from hyperbot.strategies.aggregator import AggregatedSignal


def _cfg():
    return Config(
        symbol="BTC", interval="15m", lookback=200, testnet=True,
        strategies={
            "ema_trend": StrategyConfig(True, 1.0, {"fast": 5, "slow": 10}, {}),
            "rsi_meanrev": StrategyConfig(True, 1.0, {"period": 14}, {}),
        },
        aggregator=AggregatorConfig(60, 60, 10),
        backtest=BacktestConfig(40, 20, 20, 15, 0.0005, 0.0002, 0.1, 10000.0, "total_return"),
    )


def test_analyze_returns_signals_and_aggregate():
    idx = pd.date_range("2021-01-01", periods=120, freq="15min")
    close = pd.Series(np.linspace(100, 200, 120))
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )
    sigs, agg = analyze(_cfg(), df)
    assert len(sigs) == 2
    assert isinstance(agg, AggregatedSignal)
    assert agg.decision in {"LONG", "SHORT", "FLAT"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create `hyperbot/analyze.py`**

```python
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from .config import Config
from .data_client import HyperliquidDataClient
from .strategies import REGISTRY
from .strategies.aggregator import aggregate, AggregatedSignal


def analyze(cfg: Config, df):
    signals, weights = [], {}
    for name, scfg in cfg.strategies.items():
        if not scfg.enabled:
            continue
        signals.append(REGISTRY[name](scfg.params).evaluate(df))
        weights[name] = scfg.weight
    agg = aggregate(
        signals, weights,
        cfg.aggregator.buy_threshold, cfg.aggregator.sell_threshold, cfg.aggregator.margin,
    )
    return signals, agg


def _render(signals, agg: AggregatedSignal, symbol: str, interval: str) -> Table:
    table = Table(title=f"{symbol} {interval} — read-only analysis")
    for col in ["strategy", "buy", "sell", "regime", "reason"]:
        table.add_column(col, overflow="fold")
    for s in signals:
        table.add_row(s.strategy, f"{s.buy_confidence:.0f}", f"{s.sell_confidence:.0f}", s.regime, s.reason)
    table.add_section()
    table.add_row("AGGREGATE", f"{agg.buy_confidence:.0f}", f"{agg.sell_confidence:.0f}", agg.regime, f"DECISION: {agg.decision}")
    return table


def main():
    parser = argparse.ArgumentParser(description="Live read-only market analysis (no orders).")
    parser.add_argument("--config", default="hyperbot/config.yaml")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    client = HyperliquidDataClient(testnet=cfg.testnet)
    df = client.fetch_candles(cfg.symbol, cfg.interval, cfg.lookback)
    signals, agg = analyze(cfg, df)
    Console().print(_render(signals, agg, cfg.symbol, cfg.interval))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperbot/analyze.py tests/test_analyze.py
git commit -m "feat: add live read-only analysis CLI"
```

---

## Task 16: Full-suite verification and live smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: ALL tests pass (config, indicators, base, data_client, 5 strategies, aggregator, backtest, pnl_calc, show_signals, analyze).

- [ ] **Step 2: Create local `.env` from template**

Run: `copy .env.example hyperbot\.env`
(Keys can stay empty — candle data is public.)

- [ ] **Step 3: Live read-only smoke test against testnet**

Run: `.venv\Scripts\python.exe -m hyperbot.analyze`
Expected: prints a table of 5 strategy signals plus an AGGREGATE row with a decision. If the network/SDK is unreachable, note the error — do not add retries or live-trading code.

- [ ] **Step 4: End-to-end backtest smoke test**

Run: `.venv\Scripts\python.exe -m hyperbot.backtest --out backtest_results.json`
Then: `.venv\Scripts\python.exe -m hyperbot.pnl_calc backtest_results.json`
Then: `.venv\Scripts\python.exe -m hyperbot.show_signals backtest_results.json`
Expected: backtest writes JSON; pnl_calc prints stats; show_signals prints the trade table.

- [ ] **Step 5: Confirm `.env` is gitignored**

Run: `git status --porcelain hyperbot/.env`
Expected: no output (file is ignored).

- [ ] **Step 6: Commit any final docs**

```bash
git add README.md
git commit -m "docs: finalize hyperbot README"
```

---

## Self-Review Notes

- **Spec coverage:** config.yaml + .env (Task 1–2), read-only data client (Task 5), StrategySignal + indicators (Task 3–4), 5 strategies (Task 6–10), aggregator (Task 11), walk-forward backtest (Task 12), pnl_calc/show_signals/analyze (Task 13–15). No live trading anywhere — `data_client` constructs only `Info`. ✓
- **No placeholders:** every step has runnable code/commands. ✓
- **Type consistency:** `StrategySignal(strategy, buy_confidence, sell_confidence, regime, reason, timestamp)`, `Strategy.evaluate(df)/neutral(df, reason)`, `aggregate(signals, weights, buy_threshold, sell_threshold, margin) -> AggregatedSignal`, `simulate(...) -> (trades, equity_curve, equity)`, trade dict keys consumed identically by `pnl_calc`/`show_signals`. ✓