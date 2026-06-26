from __future__ import annotations

import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"            # spot
BASE_URL_FUTURES = "https://fapi.binance.com/fapi/v1/klines"  # USDT-M perpetual
CACHE_DIR = "data/binance"
INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, session=None,
                 futures: bool = False) -> list:
    """Binance public klines, paginated forward. No auth.

    futures=True pulls the USDT-M perpetual feed (fapi, 1500/call) — the venue we trade;
    futures=False pulls spot (api, 1000/call). Same kline row format either way.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval '{interval}'. Choose from {list(INTERVAL_MS)}")
    step = INTERVAL_MS[interval]
    url = BASE_URL_FUTURES if futures else BASE_URL
    cap = 1500 if futures else 1000
    get = (session or requests).get
    rows: list = []
    cursor = start_ms
    while cursor < end_ms:
        resp = get(url, params={"symbol": symbol, "interval": interval,
                                "startTime": cursor, "endTime": end_ms, "limit": cap}, timeout=30)
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
        if len(batch) < cap:
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


def load_klines(symbol: str, interval: str, cache_dir: str = CACHE_DIR, refresh: bool = False,
                futures: bool = False) -> pd.DataFrame:
    """Full-history klines, cached to CSV. Same shape as HyperliquidDataClient.fetch_candles.

    futures=True caches the perp feed separately (e.g. BTCUSDT_perp_1h.csv).
    """
    tag = "perp_" if futures else ""
    path = os.path.join(cache_dir, f"{symbol}_{tag}{interval}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, parse_dates=["time"], index_col="time")
    start = 1_483_228_800_000          # 2017-01-01 UTC (Binance returns from listing date)
    end = int(time.time() * 1000)
    df = _rows_to_df(fetch_klines(symbol, interval, start, end, futures=futures))
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
