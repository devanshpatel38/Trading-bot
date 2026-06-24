from __future__ import annotations

import io
import os
import time
import zipfile

import pandas as pd
import requests

# Binance Data Vision archives daily USDT-M futures "metrics" (5-min granularity)
# back to 2020-09-01. Each daily ZIP holds one CSV with sum_open_interest etc.
# This is the only public source with >30 days of OI history (the live
# /futures/data/openInterestHist endpoint retains only ~30 days).
VISION_URL = "https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date}.zip"
CACHE_DIR = "data/binance"

# 30-day OI delta on an hourly grid = 720 bars.
OI_WINDOW_HOURS = 720

# Regime band edges, in percent. See classify_regime().
HIGH_FUEL = 5.0
WEAK = 2.0


def _fetch_day(symbol: str, date: str, session=None) -> pd.DataFrame | None:
    """Download+parse one daily metrics ZIP. Returns None if the day is missing (404)."""
    get = (session or requests).get
    url = VISION_URL.format(symbol=symbol, date=date)
    resp = get(url, timeout=60)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"Binance Vision {resp.status_code} for {date}: {resp.text[:200]}")
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f)
    df = df[["create_time", "sum_open_interest"]].copy()
    df["create_time"] = pd.to_datetime(df["create_time"])
    df["sum_open_interest"] = df["sum_open_interest"].astype(float)
    return df.set_index("create_time").sort_index()


def load_oi_hourly(symbol: str = "BTCUSDT", cache_dir: str = CACHE_DIR,
                   refresh: bool = False, start: str = "2020-09-01") -> pd.DataFrame:
    """Full-history hourly open interest (coin-denominated), cached to CSV.

    Resamples the 5-min source to hourly via last-value-in-hour (OI is a level/stock,
    not a flow, so the last reading is the hour's closing OI). Index is tz-naive.
    Returns a DataFrame with a single column `oi`.
    """
    path = os.path.join(cache_dir, f"{symbol}_oi_1h.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, parse_dates=["time"], index_col="time")

    session = requests.Session()
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp.utcnow().normalize().tz_localize(None)
    frames = []
    misses = 0
    day = start_date
    while day <= end_date:
        d = _fetch_day(symbol, day.strftime("%Y-%m-%d"), session)
        if d is not None:
            frames.append(d)
            misses = 0
        else:
            misses += 1
            # Tolerate a short gap, but stop after a long run of 404s (caught up to today).
            if misses > 5 and frames:
                break
        day += pd.Timedelta(days=1)
        time.sleep(0.05)  # courtesy throttle

    if not frames:
        raise RuntimeError(f"No OI metrics downloaded for {symbol}")
    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    hourly = raw["sum_open_interest"].resample("1h").last().ffill().to_frame("oi")
    os.makedirs(cache_dir, exist_ok=True)
    hourly.to_csv(path, index_label="time")
    return hourly


def classify_regime(delta_pct: float) -> str:
    """Map a 30-day OI delta (percent) to one of five regimes.

    Bands (no overlap, no gap):
      >= +5%        -> high_fuel
      +2% .. <+5%   -> weak_expansion
      -2% .. +2%    -> chop          (exclusive on both ends)
      -5% .. -2%    -> profit_taking (inclusive of -2%, exclusive of -5%)
      <= -5%        -> bleeding
    NaN (no 30d OI history yet) -> "unknown": we don't know OI is flat, so we must NOT
    treat it as chop. "unknown" is never tradable; the backtester stands aside on it.
    This matters live, where only a recent OI window is fetched.
    """
    if delta_pct is None or pd.isna(delta_pct):
        return "unknown"
    if delta_pct >= HIGH_FUEL:
        return "high_fuel"
    if delta_pct >= WEAK:
        return "weak_expansion"
    if delta_pct > -WEAK:
        return "chop"
    if delta_pct > -HIGH_FUEL:
        return "profit_taking"
    return "bleeding"


def oi_delta_on_index(candle_index: pd.DatetimeIndex, oi_hourly: pd.DataFrame,
                      window: int = OI_WINDOW_HOURS, avg_hours: int | None = 24) -> pd.Series:
    """OI delta (%) aligned to the candle index, computed causally on the candle grid.

    Reindexes hourly OI onto the candle timestamps (forward-fill), then takes the
    percent change over `window` bars. On a contiguous 1h grid, window=720 == 30 days.

    avg_hours: when set (e.g. 24), smooth both endpoints instead of comparing two single
    bars: compare the *trailing* `avg_hours` average ending at t against the *centered*
    `avg_hours` average around t-window. This removes the regime's sensitivity to a noisy
    OI spike on the single reference bar 30 days ago. Both windows are causal (the centered
    reference sits ~window hours in the past, so its future half is still historical).
    """
    oi = oi_hourly["oi"].reindex(candle_index, method="ffill")
    if not avg_hours:
        return (oi / oi.shift(window) - 1.0) * 100.0
    trailing = oi.rolling(avg_hours).mean()                     # avg of [t-avg_hours+1 .. t]
    reference = oi.rolling(avg_hours, center=True).mean().shift(window)  # avg around t-window
    return (trailing / reference - 1.0) * 100.0


def regime_series(candle_index: pd.DatetimeIndex, oi_hourly: pd.DataFrame,
                  window: int = OI_WINDOW_HOURS, avg_hours: int | None = 24) -> pd.Series:
    """Regime label per candle bar (str Series aligned to candle_index)."""
    delta = oi_delta_on_index(candle_index, oi_hourly, window, avg_hours)
    return delta.map(classify_regime)


def fetch_recent_oi_hourly(symbol: str = "BTCUSDT", days: int = 50) -> pd.DataFrame:
    """Hourly OI for roughly the last `days` days, fetched fresh (no cache).

    Used by the live notifier: a 30-day smoothed delta needs ~31 days of history, so a
    ~50-day window covers it plus a buffer for any recently-opened trade. Pulls the small
    daily Binance Vision metrics files (~50 requests, a few seconds) — far cheaper than the
    multi-year `load_oi_hourly`, and the live OI-history API only retains ~30 days.
    """
    session = requests.Session()
    end_date = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start_date = end_date - pd.Timedelta(days=days)
    frames = []
    day = start_date
    while day <= end_date:
        d = _fetch_day(symbol, day.strftime("%Y-%m-%d"), session)
        if d is not None:
            frames.append(d)
        day += pd.Timedelta(days=1)
    if not frames:
        raise RuntimeError(f"No recent OI metrics downloaded for {symbol}")
    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw["sum_open_interest"].resample("1h").last().ffill().to_frame("oi")