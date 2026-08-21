"""Backtest-parity: the live bot must NOT re-enter on the candle a trade closed on
(the engine closes then `continue`s, never entering a new trade on the exit bar)."""
import hyperbot.binance_bot as bot


class FakeClient:
    def __init__(self, positions):
        self._positions = list(positions)
        self._i = 0
        self.market_calls = []

    def position(self):
        p = self._positions[min(self._i, len(self._positions) - 1)]
        self._i += 1
        return p

    def open_algo_orders(self):
        return []

    def cancel_all(self):
        pass

    def available_usdt(self):
        return 5000.0

    def round_qty(self, q):
        return round(q, 3)

    def round_price(self, p):
        return round(p, 1)

    def market(self, side, qty):
        self.market_calls.append((side, qty))
        return {"status": "FILLED", "orderId": 1}

    def stop_market(self, *a, **k):
        return {"algoId": 10}

    def take_profit(self, *a, **k):
        return {"algoId": 11}


def _wire(monkeypatch, tmp_path, fake, bar="2026-08-21 08:00:00"):
    monkeypatch.setattr(bot, "BinanceFuturesClient", lambda *a, **k: fake)
    monkeypatch.setattr(bot, "latest_closed_bar", lambda *a, **k: bar)
    monkeypatch.setattr(bot, "evaluate_signal", lambda cfg: {
        "signal": "long", "bar": bar, "ref_price": 100.0,
        "stop_dist": 5.0, "stop": 95.0, "tp": 115.0, "regime": "chop"})
    monkeypatch.setattr(bot, "STATE_PATH", tmp_path / "state.json")


def test_no_reentry_on_close_candle(tmp_path, monkeypatch):
    IN_POS = {"side": "long", "qty": 0.1, "entry": 100.0, "mark": 101.0, "unreal": 1.0}
    # run1: in position ; run2: flat (trade just closed) ; run3: still flat
    fake = FakeClient([IN_POS, None, None])
    _wire(monkeypatch, tmp_path, fake)

    bot.run_once(testnet=True)                     # in position -> no entry
    assert fake.market_calls == []
    assert bot.load_state()["in_position"] is True

    bot.run_once(testnet=True)                     # flat, just closed -> SKIP re-entry (parity)
    assert fake.market_calls == []
    assert bot.load_state()["in_position"] is False

    bot.run_once(testnet=True)                     # flat, next candle -> ENTER
    assert len(fake.market_calls) == 1
    assert bot.load_state()["in_position"] is True


def test_flat_with_signal_enters_immediately(tmp_path, monkeypatch):
    # Never held a position -> a fresh flat run with a signal enters right away.
    fake = FakeClient([None])
    _wire(monkeypatch, tmp_path, fake)
    bot.run_once(testnet=True)
    assert len(fake.market_calls) == 1
