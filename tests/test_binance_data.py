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
