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
