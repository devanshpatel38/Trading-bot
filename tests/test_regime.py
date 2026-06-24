"""Tests for OI-regime classification, regime-aware voting, and trade-management math."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from hyperbot.oi_data import classify_regime, oi_delta_on_index, regime_series
from hyperbot.strategies.aggregator import aggregate_regime
from hyperbot.strategies.base import StrategySignal
from hyperbot.backtest import _close_trade


# --------------------------------------------------------------------------- #
# Regime classification boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta,expected", [
    (10.0, "high_fuel"),
    (5.0, "high_fuel"),       # inclusive lower edge
    (4.99, "weak_expansion"),
    (2.0, "weak_expansion"),  # inclusive lower edge
    (1.99, "chop"),
    (0.0, "chop"),
    (-1.99, "chop"),
    (-2.0, "profit_taking"),  # inclusive
    (-4.99, "profit_taking"),
    (-5.0, "bleeding"),       # inclusive
    (-10.0, "bleeding"),
])
def test_classify_regime_boundaries(delta, expected):
    assert classify_regime(delta) == expected


def test_classify_regime_nan_is_unknown():
    # No 30d OI history -> "unknown" (never tradable), not a loose "chop".
    assert classify_regime(float("nan")) == "unknown"
    assert classify_regime(None) == "unknown"


# --------------------------------------------------------------------------- #
# OI delta computation (causal, on candle grid)
# --------------------------------------------------------------------------- #
def test_oi_delta_on_index():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h")
    oi = pd.DataFrame({"oi": [100.0, 110.0, 120.0, 90.0, 100.0]}, index=idx)
    delta = oi_delta_on_index(idx, oi, window=2, avg_hours=None)
    # bar 0,1 -> NaN (no t-2). bar2: 120/100-1=20%. bar3: 90/110-1=-18.18%. bar4: 100/120-1=-16.67%
    assert math.isnan(delta.iloc[0]) and math.isnan(delta.iloc[1])
    assert delta.iloc[2] == pytest.approx(20.0)
    assert delta.iloc[3] == pytest.approx(-18.1818, abs=1e-3)
    assert delta.iloc[4] == pytest.approx(-16.6667, abs=1e-3)


def test_oi_delta_smoothed_reference():
    # Linear OI, slope 1. avg_hours=3 (odd -> unambiguous centered window), window=3.
    # trailing(3)[t] = oi[t]-1 ; centered(3)[j] = oi[j] ; reference[t] = oi[t-3].
    idx = pd.date_range("2024-01-01", periods=12, freq="1h")
    oi = pd.DataFrame({"oi": [100.0 + i for i in range(12)]}, index=idx)
    delta = oi_delta_on_index(idx, oi, window=3, avg_hours=3)
    # t=8: trailing=mean(106,107,108)=107, reference=oi[5]=105 -> (107/105-1)*100
    assert delta.iloc[8] == pytest.approx((107.0 / 105.0 - 1) * 100, abs=1e-6)


def test_regime_series_maps_labels():
    idx = pd.date_range("2024-01-01", periods=4, freq="1h")
    oi = pd.DataFrame({"oi": [100.0, 100.0, 106.0, 90.0]}, index=idx)
    reg = regime_series(idx, oi, window=1, avg_hours=None)
    # bar0 NaN->unknown, bar1 0%->chop, bar2 +6%->high_fuel, bar3 -15%->bleeding
    assert list(reg) == ["unknown", "chop", "high_fuel", "bleeding"]


# --------------------------------------------------------------------------- #
# Regime-aware voting
# --------------------------------------------------------------------------- #
def sig(name, buy=0.0, sell=0.0):
    return StrategySignal(name, buy, sell, "x", "", None)


def all_sigs(buy_names=(), sell_names=()):
    names = ["ema_trend", "macd_momentum", "bb_squeeze", "rsi_meanrev", "fvg"]
    out = []
    for nm in names:
        out.append(sig(nm, 75.0 if nm in buy_names else 0.0, 75.0 if nm in sell_names else 0.0))
    return out


def test_high_fuel_needs_unanimous():
    five = ("ema_trend", "macd_momentum", "bb_squeeze", "rsi_meanrev", "fvg")
    rec, agreed = aggregate_regime(all_sigs(buy_names=five), "high_fuel")
    assert rec == "long"
    assert set(agreed) == set(five)
    # only 4 agree -> stand aside (high fuel is back to strict 5/5)
    rec2, _ = aggregate_regime(all_sigs(buy_names=five[:4]), "high_fuel")
    assert rec2 == "stand_aside"


def test_weak_expansion_needs_four_incl_core():
    rec, _ = aggregate_regime(all_sigs(buy_names=("ema_trend", "bb_squeeze", "rsi_meanrev", "fvg")), "weak_expansion")
    assert rec == "long"
    # only 3 agree -> stand aside
    rec2, _ = aggregate_regime(all_sigs(buy_names=("bb_squeeze", "rsi_meanrev", "fvg")), "weak_expansion")
    assert rec2 == "stand_aside"


def test_chop_and_profit_taking_need_unanimous():
    five = ("ema_trend", "macd_momentum", "bb_squeeze", "rsi_meanrev", "fvg")
    for regime in ("chop", "profit_taking"):
        assert aggregate_regime(all_sigs(buy_names=five), regime)[0] == "long"
        assert aggregate_regime(all_sigs(buy_names=five[:4]), regime)[0] == "stand_aside"


def test_bleeding_needs_both_mr():
    rec, agreed = aggregate_regime(all_sigs(sell_names=("rsi_meanrev", "fvg")), "bleeding")
    assert rec == "short"
    assert set(agreed) == {"rsi_meanrev", "fvg"}
    # only one MR -> stand aside; trend votes ignored
    assert aggregate_regime(all_sigs(sell_names=("rsi_meanrev", "ema_trend", "macd_momentum", "bb_squeeze")), "bleeding")[0] == "stand_aside"


# --------------------------------------------------------------------------- #
# Trade-management R accounting (_close_trade)
# --------------------------------------------------------------------------- #
def make_ot(*, side="long", entry=100.0, stop_dist=10.0, rr=3.0,
            scaled=False, partial_frac=0.0, partial_level_r=0.0, partial_exit=None,
            be_moved=False):
    sign = 1.0 if side == "long" else -1.0
    return {
        "entry": entry, "stop_dist": stop_dist, "rr": rr,
        "stop": entry - sign * stop_dist, "tp": entry + sign * rr * stop_dist,
        "scaled": scaled, "partial_frac": partial_frac, "partial_level_r": partial_level_r,
        "partial_exit": partial_exit, "be_moved": be_moved, "_entry_i": 0,
    }


def test_single_leg_win():
    ot = make_ot()
    _close_trade(ot, ot["tp"], 5, pd.date_range("2024-01-01", periods=6, freq="1h"), 0.0, 0.0)
    assert ot["outcome"] == "win"
    assert ot["r_multiple"] == pytest.approx(3.0)


def test_single_leg_loss():
    ot = make_ot()
    _close_trade(ot, ot["stop"], 5, pd.date_range("2024-01-01", periods=6, freq="1h"), 0.0, 0.0)
    assert ot["outcome"] == "loss"
    assert ot["r_multiple"] == pytest.approx(-1.0)


def test_scaled_runner_to_tp():
    # 50% out at +2R, runner 50% to +3R -> 0.5*2 + 0.5*3 = 2.5R
    ot = make_ot(scaled=True, partial_frac=0.5, partial_level_r=2.0, partial_exit=120.0)
    _close_trade(ot, ot["tp"], 5, pd.date_range("2024-01-01", periods=6, freq="1h"), 0.0, 0.0)
    assert ot["r_multiple"] == pytest.approx(2.5)


def test_scaled_runner_to_stop():
    # 50% out at +2R, runner 50% hits original stop -> 0.5*2 + 0.5*(-1) = 0.5R
    ot = make_ot(scaled=True, partial_frac=0.5, partial_level_r=2.0, partial_exit=120.0)
    _close_trade(ot, ot["stop"], 5, pd.date_range("2024-01-01", periods=6, freq="1h"), 0.0, 0.0)
    assert ot["r_multiple"] == pytest.approx(0.5)


def test_breakeven_exit_is_flat():
    # stop moved to entry, exit at entry -> 0 gross
    ot = make_ot(be_moved=True)
    ot["stop"] = ot["entry"]
    _close_trade(ot, ot["entry"], 5, pd.date_range("2024-01-01", periods=6, freq="1h"), 0.0, 0.0)
    assert ot["gross_r"] == pytest.approx(0.0)


def test_scaled_then_breakeven_runner():
    # 50% banked at +2R, then runner stopped at breakeven -> 0.5*2 + 0.5*0 = +1.0R
    ot = make_ot(scaled=True, partial_frac=0.5, partial_level_r=2.0, partial_exit=120.0, be_moved=True)
    ot["stop"] = ot["entry"]
    _close_trade(ot, ot["entry"], 5, pd.date_range("2024-01-01", periods=6, freq="1h"), 0.0, 0.0)
    assert ot["r_multiple"] == pytest.approx(1.0)