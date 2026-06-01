import hyperbot.data_client as dc


def test_fetch_candles_builds_dataframe(monkeypatch):
    raw = [
        {"t": 1700000000000, "o": "1.0", "h": "2.0", "l": "0.5", "c": "1.5", "v": "100"},
        {"t": 1700000900000, "o": "1.5", "h": "2.5", "l": "1.0", "c": "2.0", "v": "150"},
    ]

    class FakeInfo:
        def __init__(self, *args, **kwargs):
            pass

        def candles_snapshot(self, name, interval, startTime, endTime):
            return raw

    monkeypatch.setattr(dc, "Info", FakeInfo)
    client = dc.HyperliquidDataClient(testnet=True)
    df = client.fetch_candles("BTC", "15m", lookback=2)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[-1] == 2.0
    assert df.index.is_monotonic_increasing


def test_fetch_candles_days_paginates(monkeypatch):
    step = dc.INTERVAL_MS["15m"]
    # build 7 synthetic candles spaced by `step`
    base = 1_700_000_000_000
    allc = [{"t": base + k * step, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"} for k in range(7)]

    class FakeInfo:
        def __init__(self, *a, **k):
            pass

        def candles_snapshot(self, coin, interval, start, end):
            window = [c for c in allc if start <= c["t"] <= end]
            return window[-3:]  # API returns only the most recent <=3 in range

    monkeypatch.setattr(dc, "Info", FakeInfo)
    client = dc.HyperliquidDataClient(testnet=True)
    df = client.fetch_candles_days("BTC", "15m", days=9999)  # large window so start<=base
    assert len(df) == 7
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique


def test_fetch_candles_raises_on_empty(monkeypatch):
    class EmptyInfo:
        def __init__(self, *args, **kwargs):
            pass

        def candles_snapshot(self, *a, **k):
            return []

    monkeypatch.setattr(dc, "Info", EmptyInfo)
    client = dc.HyperliquidDataClient(testnet=True)
    try:
        client.fetch_candles("BTC", "15m", 2)
        assert False, "expected ValueError"
    except ValueError:
        pass
