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