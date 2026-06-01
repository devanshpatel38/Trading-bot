import pandas as pd
import pytest

from hyperbot.backtest import run_backtest, summarize, attribution
from hyperbot.strategies.base import Strategy, StrategySignal


def _df(rows):  # rows: list of (o,h,l,c)
    idx = pd.date_range("2021-01-01", periods=len(rows), freq="15min")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": 100.0},
        index=idx,
    )


class _StubLong(Strategy):
    name = "stub"
    def __init__(self, fire_at, seen):
        super().__init__(); self.fire_at = fire_at; self.seen = seen
    def analyze(self, df):
        i = len(df) - 1
        self.seen.append(i)               # record the largest index visible -> no-lookahead check
        if i == self.fire_at:
            return StrategySignal(self.name, 100.0, 0.0, "trending", "fire", df.index[-1])
        return StrategySignal(self.name, 0.0, 0.0, "ranging", "flat", df.index[-1])


def _run(rows, fire_at, warmup=2, rr=2.0, fee=0.0, slippage=0.0):
    seen = []
    trades = run_backtest(_df(rows), {"stub": _StubLong(fire_at, seen)},
                          threshold=50, min_agree=1, margin=15, rr=rr,
                          atr_period=2, atr_mult=1.0, warmup=warmup,
                          fee=fee, slippage=slippage)
    return trades, seen


def test_long_tp_hit_is_win():
    # flat bars, fire at index 3 (entry=close=100, ATR(2)=2 -> tp=104). next bar high 130 -> TP.
    rows = [(100,101,99,100)]*3 + [(100,101,99,100)]   # entry bar i=3 close 100
    rows += [(100, 130, 100, 120)]                      # next bar high spikes -> TP
    trades, _ = _run(rows, fire_at=3)
    assert len(trades) == 1
    assert trades[0]["side"] == "long"
    assert trades[0]["outcome"] == "win"
    assert trades[0]["r_multiple"] == 2.0
    assert trades[0]["bars_held"] == 1

def test_costs_reduce_r():
    # same winning-long scenario as test_long_tp_hit_is_win, but with fee + slippage.
    rows = [(100,101,99,100)]*3 + [(100,101,99,100)]
    rows += [(100, 130, 100, 120)]
    trades, _ = _run(rows, fire_at=3, fee=0.001, slippage=0.001)
    t = trades[0]
    assert t["outcome"] == "win"
    assert t["gross_r"] == 2.0
    stop_dist = abs(t["entry"] - t["stop"])
    expected_cost = round(0.002 * (t["entry"] + t["tp"]) / stop_dist, 4)
    assert t["cost_r"] == expected_cost
    assert t["r_multiple"] == round(2.0 - expected_cost, 4)
    assert t["r_multiple"] < 2.0

def test_long_stop_hit_is_loss():
    rows = [(100,101,99,100)]*3 + [(100,101,99,100)]
    rows += [(100, 101, 70, 80)]                        # next bar low crashes -> stop (stop=98)
    trades, _ = _run(rows, fire_at=3)
    assert trades[0]["outcome"] == "loss"
    assert trades[0]["r_multiple"] == -1.0

def test_both_hit_in_bar_is_loss():
    rows = [(100,101,99,100)]*3 + [(100,101,99,100)]
    rows += [(100, 130, 70, 100)]                       # both TP and stop inside the bar
    trades, _ = _run(rows, fire_at=3)
    assert trades[0]["outcome"] == "loss"

def test_open_at_end_excluded_from_winrate():
    rows = [(100,101,99,100)]*3 + [(100,101,99,100), (100,101,99,100)]
    trades, _ = _run(rows, fire_at=3)
    assert trades[0]["outcome"] == "open"
    assert summarize(trades)["resolved"] == 0

def test_no_lookahead_window_grows_by_one():
    rows = [(100,101,99,100)]*8
    _, seen = _run(rows, fire_at=99)  # never fires
    assert seen == list(range(2, 8))  # warmup=2 .. n-1, strictly one bar at a time

def test_attribution_counts_agreement_on_wins():
    rows = [(100,101,99,100)]*3 + [(100,101,99,100), (100,130,100,120)]
    trades, _ = _run(rows, fire_at=3)
    attr = attribution(trades, ["stub"])
    assert attr["stub"]["agreed_wins"] == 1
    assert attr["stub"]["win_rate_when_agreed"] == 100.0