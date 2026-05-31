# Hyperbot v2 — PDF-aligned strategies & aggregator

> **For agentic workers:** executed task-by-task via superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes. Each STEP must end with the full suite green (`.venv\Scripts\python.exe -m pytest -q`). Commit with plain `git commit` (local identity already set). Do NOT push.

**Goal:** Align the 5 strategies and the aggregator to `Trading_Model_Documented.pdf` (pp. 8–10): rename `evaluate`→`analyze`, score every strategy as **four binary 0/25 components** (sum 0–100, traceable in `reason`), and replace the aggregator with the doc's agreement-count design.

**Key rule:** Each of a strategy's 4 components contributes **exactly 0 or 25** (never graduated). `buy_confidence = 25 × (#buy components true)`, same for sell. `reason` must list each component's score.

---

## STEP 1 — Rename `evaluate` → `analyze` (mechanical)

**Files:** `hyperbot/strategies/base.py`, all 5 strategy modules, `hyperbot/backtest.py`, `hyperbot/analyze.py`, and every `tests/test_*.py` that calls `.evaluate(`.

- [ ] In `base.py` `Strategy` ABC, rename the abstract method `evaluate` → `analyze` (keep signature `analyze(self, df) -> StrategySignal`). `neutral()` stays.
- [ ] In each strategy module rename `def evaluate` → `def analyze`.
- [ ] In `backtest.py` `simulate()`, change `s.evaluate(window)` → `s.analyze(window)`.
- [ ] In `analyze.py`, change `.evaluate(df)` → `.analyze(df)`. (Note: module-level function is also named `analyze`; the method call is `REGISTRY[name](scfg.params).analyze(df)` — both can coexist.)
- [ ] In tests, replace all `.evaluate(` → `.analyze(`.
- [ ] Run full suite — expect all green (33 tests). Commit: `refactor: rename Strategy.evaluate to analyze`.

---

## STEP 2 — Agreement-based aggregator + config + backtest rewiring

### 2a. `config.py` — `AggregatorConfig`

Replace fields `buy_threshold, sell_threshold, margin` with:
```python
@dataclass
class AggregatorConfig:
    threshold: float    # confidence required to "agree" (50)
    min_agree: int      # min strategies agreeing (3)
    margin: float       # agreed avg must beat opposite by this (15)
```

### 2b. `config.yaml`

Replace the `aggregator:` block with:
```yaml
aggregator:
  threshold: 50
  min_agree: 3
  margin: 15
```
And bump backtest warmup so EMA200 is meaningful:
```yaml
backtest:
  ...
  warmup_bars: 220
```

### 2c. `hyperbot/strategies/aggregator.py` — full replacement

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AggregatedSignal:
    recommendation: str  # "long" | "short" | "stand_aside"
    avg_buy: float
    avg_sell: float
    agree_buy: int
    agree_sell: int
    regime: str
    reason: str
    components: list = field(default_factory=list)


def aggregate(signals, threshold: float = 50.0, min_agree: int = 3, margin: float = 15.0) -> AggregatedSignal:
    if not signals:
        return AggregatedSignal("stand_aside", 0.0, 0.0, 0, 0, "unknown", "no signals", [])
    n = len(signals)
    agree_buy = sum(1 for s in signals if s.buy_confidence >= threshold)
    agree_sell = sum(1 for s in signals if s.sell_confidence >= threshold)
    avg_buy = sum(s.buy_confidence for s in signals) / n
    avg_sell = sum(s.sell_confidence for s in signals) / n
    if agree_buy >= min_agree and avg_buy >= threshold * 0.8 and avg_buy > avg_sell + margin:
        rec = "long"
    elif agree_sell >= min_agree and avg_sell >= threshold * 0.8 and avg_sell > avg_buy + margin:
        rec = "short"
    else:
        rec = "stand_aside"
    regime_scores: dict[str, int] = {}
    for s in signals:
        regime_scores[s.regime] = regime_scores.get(s.regime, 0) + 1
    regime = max(regime_scores.items(), key=lambda kv: kv[1])[0]
    reason = (
        f"agree_buy={agree_buy} agree_sell={agree_sell} "
        f"avg_buy={avg_buy:.1f} avg_sell={avg_sell:.1f} -> {rec}"
    )
    return AggregatedSignal(rec, round(avg_buy, 4), round(avg_sell, 4), agree_buy, agree_sell, regime, reason, list(signals))
```

### 2d. `backtest.py` — deciders + `simulate` signature

`simulate` now takes a `decide(signals) -> AggregatedSignal` callable instead of `(weights, agg_cfg)`:

```python
def simulate(df, strategies, decide, bt_cfg, start_offset, initial_equity):
    # ... unchanged equity/position bookkeeping ...
    for i in range(start_offset, len(df)):
        window = df.iloc[: i + 1]
        sigs = [s.analyze(window) for s in strategies.values()]
        agg = decide(sigs)
        price = float(closes[i]); t = index[i]
        target = 1 if agg.recommendation == "long" else (-1 if agg.recommendation == "short" else 0)
        # on open, record from agg: buy_confidence=agg.avg_buy, sell_confidence=agg.avg_sell,
        #   regime=agg.regime, entry_reason=agg.reason   (keep these exact trade-dict keys)
        ...
```
Keep the trade-dict keys identical (`buy_confidence`, `sell_confidence`, `regime`, `entry_reason`, etc.) so `pnl_calc`/`show_signals` need no change. Use `agg.avg_buy`/`agg.avg_sell` for those fields.

Add decider builders and update `optimize`/`walk_forward`:
```python
from .strategies.aggregator import aggregate, AggregatedSignal

def make_aggregator_decider(cfg):
    a = cfg.aggregator
    return lambda sigs: aggregate(sigs, a.threshold, a.min_agree, a.margin)

def make_solo_decider(threshold, margin):
    def decide(sigs):
        s = sigs[0]
        if s.buy_confidence >= threshold and s.buy_confidence > s.sell_confidence + margin:
            rec = "long"
        elif s.sell_confidence >= threshold and s.sell_confidence > s.buy_confidence + margin:
            rec = "short"
        else:
            rec = "stand_aside"
        return AggregatedSignal(rec, s.buy_confidence, s.sell_confidence,
                                int(s.buy_confidence >= threshold), int(s.sell_confidence >= threshold),
                                s.regime, s.reason, [s])
    return decide
```
- `optimize(...)`: each strategy is scored solo — call
  `simulate(is_df, {name: strat}, make_solo_decider(cfg.aggregator.threshold, cfg.aggregator.margin), _bt_cfg(cfg), cfg.backtest.warmup_bars, cfg.backtest.initial_equity)`.
  (A single strategy can never reach `min_agree=3`, so the OOS aggregator can't be used for solo optimization — this solo decider is why.)
- `walk_forward(...)`: build `decide = make_aggregator_decider(cfg)` once and pass it to the OOS `simulate(...)`. Drop the old `weights`/`_agg_cfg` plumbing. `_agg_cfg` helper can be removed.

### 2e. `analyze.py`

`analyze(cfg, df)` keeps returning `(signals, agg)` but call `aggregate(signals, cfg.aggregator.threshold, cfg.aggregator.min_agree, cfg.aggregator.margin)`. In `_render`, replace the aggregate row to print `agg.avg_buy`, `agg.avg_sell`, and `DECISION: {agg.recommendation}`.

### 2f. Tests

- `tests/test_aggregator.py` — rewrite for the new design:
  - 5 signals where ≥3 have `buy_confidence>=50` and avg_buy>avg_sell+15 → `recommendation=="long"`.
  - Only 2 agree → `stand_aside`.
  - Mirror case → `short`.
  - `aggregate([])` → `stand_aside`.
- `tests/test_backtest.py` — update `test_simulate_long_in_uptrend_profits` to pass a decider, e.g.
  `from hyperbot.backtest import make_solo_decider` and call
  `simulate(df, {"always_long": _AlwaysLong()}, make_solo_decider(50, 10), bt_cfg, 5, 10000.0)`.
  `_AlwaysLong.analyze` returns buy 100 → solo decider → "long". The `walk_forward` smoke test: ensure the single-strategy config still produces the result dict (with `min_agree=3` the OOS aggregator yields `stand_aside`, so 0 trades is acceptable — assert structure keys and `windows` length, not trade count).
- [ ] Run full suite — all green. Commit: `feat: agreement-based aggregator (>=3 of 5) per PDF spec`.

---

## STEP 3 — Refactor the 5 strategies to 4×25 binary scoring

For every strategy: `analyze(df)` returns `StrategySignal(name, buy, sell, regime, reason, last_timestamp(df))` where `buy = 25*sum(buy_components)` and `sell = 25*sum(sell_components)`. Guard insufficient data → `self.neutral(df, "insufficient data")`. `reason` lists each component, e.g. `"htf=25 align=25 pullback=0 candle=25 -> buy 75"`. Update each strategy's `default_params` and the matching `config.yaml` block + grid. Rewrite each strategy's test to assert the new binary behavior.

### 3.1 EMA Trend Pullback — `ema_trend.py`
`default_params = {"fast": 20, "slow": 200, "atr_period": 14, "pullback_atr": 0.5}`
Need `len(df) >= slow + 1` (≥201). Compute EMA20, EMA200, ATR; `c=close[-1]`, `o=open[-1]`.
Buy components (each 0/25):
1. HTF trend: `c > EMA200[-1]`
2. EMA alignment: `EMA20[-1] > EMA200[-1]`
3. Pullback zone: `abs(c - EMA20[-1]) <= 0.5*ATR[-1]`
4. Candle direction: `c > o`
Sell components: mirror (`c < EMA200`, `EMA20 < EMA200`, same pullback zone, `c < o`).
regime: `"trending"` if alignment holds (EMA20 vs EMA200 separated) else `"ranging"`.
config grid: `{pullback_atr: [0.5, 1.0]}`.
Tests: a constructed strong uptrend with price near EMA20 and bullish last candle → `buy >= 75`, `buy > sell`; insufficient data (<201 bars) → neutral 0/0.

### 3.2 RSI Mean Reversion — `rsi_meanrev.py`
`default_params = {"period": 14, "oversold": 30.0, "overbought": 70.0, "ema_period": 50, "atr_period": 14, "mean_atr": 1.5}`
Need `len(df) >= ema_period + 1` (≥51). Compute RSI, EMA50, ATR; `r=RSI[-1]`, `r_prev=RSI[-2]`, `c, o`.
Buy components:
1. RSI extreme: `r < oversold`
2. RSI direction (turning toward neutral, i.e. up): `r > r_prev`
3. Mean proximity: `abs(c - EMA50[-1]) <= mean_atr*ATR[-1]`
4. Candle direction: `c > o`
Sell components: `r > overbought`, `r < r_prev`, same proximity, `c < o`.
regime: `"ranging"`.
config grid: `{oversold: [25, 30]}`.
Tests: a series ending oversold + RSI ticking up + bullish candle near EMA50 → `buy >= 75`; insufficient data → neutral.

### 3.3 Bollinger Band Squeeze — `bb_squeeze.py`
`default_params = {"period": 20, "num_std": 2.0, "squeeze_lookback": 50, "squeeze_quantile": 0.25, "vol_lookback": 20}`
Need `len(df) >= period + squeeze_lookback`. Compute bands, `bw=(upper-lower)/mid`, `vol_ma = volume.rolling(vol_lookback).mean()`.
`bw_thresh = bw.iloc[-squeeze_lookback:].quantile(squeeze_quantile)`.
Shared components (count for the breakout direction):
1. Squeeze regime: `bw.iloc[-2] <= bw_thresh` (was compressed on prior bar)
2. Expansion: `bw.iloc[-1] > bw.iloc[-2]`
3. Breakout direction: buy = `c > upper[-1]`; sell = `c < lower[-1]`
4. Volume confirmation: `volume[-1] > vol_ma[-1]`
Buy = 25×(comp1,2,4 true **and** buy-direction comp3); i.e. components 1,2,4 are shared, component 3 is the directional one. Concretely: `buy_comps = [squeeze, expansion, c>upper, vol_ok]`, `sell_comps = [squeeze, expansion, c<lower, vol_ok]`.
regime: `"squeeze"` if comp1 and not breaking out; `"expansion"` if breaking out; else `"ranging"`.
config grid: `{period: [20, 30]}`.
Tests: build ≥70 bars: long low-volatility flat section (squeeze) then a final bar that closes above the upper band on above-average volume with rising bandwidth → `buy >= 75`; insufficient data → neutral.

### 3.4 Fair Value Gap — `fvg.py`
`default_params = {"atr_period": 14, "proximity_atr": 0.25, "fvg_lookback": 20}`
Need `len(df) >= max(fvg_lookback, atr_period) + 3`. `a = ATR[-1]` (guard NaN/≤0 → neutral). `c = close[-1]`.
Scan the most recent `fvg_lookback` bars for the **latest** FVG using indices (j-2, j-1, j):
- bullish gap at j if `low[j] > high[j-2]`; zone = `(high[j-2], low[j])` (bottom, top).
- bearish gap at j if `high[j] < low[j-2]`; zone = `(high[j], low[j-2])`.
Take the most recent such gap (largest j). Components for that gap's direction:
1. Gap exists: a qualifying FVG was found in the window.
2. Unfilled (close-based): no close **after** the gap (bars j+1..-1) has re-entered the zone — bullish filled if any later `close < zone_top`; bearish filled if any later `close > zone_bottom`. Unfilled = not filled.
3. Proximity gate: current `c` within `proximity_atr*a` of the zone (distance to nearest zone edge ≤ `0.25*a`; 0 if inside).
4. Direction confirmation: bullish gap → `c > open[-1]`; bearish gap → `c < open[-1]`.
If a bullish gap is latest, fill `buy_comps`; if bearish, fill `sell_comps`. The opposite side is 0.
regime: `"imbalance"` if a gap exists else `"balanced"`.
config grid: `{proximity_atr: [0.25, 0.5]}`.
Tests: construct a recent unfilled bullish FVG with current price within 0.25 ATR and a bullish last candle → `buy >= 75`, `sell == 0`; a series with no gap → neutral 0/0, regime `"balanced"`; insufficient data → neutral.

### 3.5 MACD Momentum — `macd_momentum.py`
`default_params = {"fast": 12, "slow": 26, "signal": 9, "ema_period": 200, "cross_lookback": 3}`
Need `len(df) >= ema_period + 1` (≥201). Compute `macd_line, signal_line, hist`, EMA200; `m=macd_line[-1]`, `c=close[-1]`.
Fresh crossover within last `cross_lookback` bars: bullish if for any k in 1..cross_lookback, `macd_line[-k-1] <= signal_line[-k-1]` and `macd_line[-k] > signal_line[-k]`; bearish mirror.
Buy components:
1. Fresh crossover: bullish cross within last 3 bars
2. Histogram acceleration: `hist[-1] > hist[-2]`
3. EMA200 HTF filter: `c > EMA200[-1]`
4. Zero-line position: `m > 0`
Sell components: bearish cross within last 3 bars; `hist[-1] < hist[-2]`; `c < EMA200[-1]`; `m < 0`.
regime: `"trending"` if EMA200 filter + crossover align, else `"ranging"`.
config grid: `{fast: [8, 12]}`.
Tests: a sustained uptrend that produced a recent bullish cross above EMA200 with rising histogram and m>0 → `buy >= 75`; insufficient data (<201) → neutral.

- [ ] After all 5 refactored + tests rewritten + `config.yaml` strategy blocks/grids updated: run full suite — all green. Commit per strategy (or one commit `feat: 4x25 binary scoring for all strategies per PDF`).

---

## Self-review checklist
- `analyze` everywhere; no `.evaluate(` remains (`grep`).
- Every strategy confidence ∈ {0,25,50,75,100}; `reason` enumerates components.
- Aggregator returns `long/short/stand_aside`; `simulate` maps it; trade-dict keys unchanged so `pnl_calc`/`show_signals` untouched.
- `optimize` uses the solo decider; `walk_forward` uses the aggregator decider.
- EMA200/EMA50 data guards correct; `warmup_bars=220`.
- Full suite green.