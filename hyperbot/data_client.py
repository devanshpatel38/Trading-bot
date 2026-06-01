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
        raw = self.info.candles_snapshot(symbol, interval, start, end)
        if not raw:
            raise ValueError(f"No candles returned for {symbol} {interval}")
        df = pd.DataFrame(raw).rename(
            columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.set_index("time")[["open", "high", "low", "close", "volume"]].sort_index()

    def fetch_candles_days(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch `days` of candles, paginating backward (API caps ~5000/call)."""
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported interval '{interval}'. Choose from {list(INTERVAL_MS)}")
        step = INTERVAL_MS[interval]
        end = int(time.time() * 1000)
        start = end - days * 86_400_000
        rows: dict[int, dict] = {}
        cursor_end = end
        while cursor_end > start:
            batch = self.info.candles_snapshot(symbol, interval, start, cursor_end)
            if not batch:
                break
            for c in batch:
                rows[int(c["t"])] = c
            earliest = min(int(c["t"]) for c in batch)
            if earliest <= start or earliest >= cursor_end:
                break  # covered the window, or no backward progress
            cursor_end = earliest - step
        if not rows:
            raise ValueError(f"No candles returned for {symbol} {interval}")
        ordered = [rows[t] for t in sorted(rows) if t >= start]
        df = pd.DataFrame(ordered).rename(
            columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.set_index("time")[["open", "high", "low", "close", "volume"]].sort_index()