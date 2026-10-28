"""Download and preparation of the public datasets.

Nothing here is needed by the tests; they run on synthetic fixtures.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import paths
from .dq import dst_transitions

UCI_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
SGCC_BASE = "https://raw.githubusercontent.com/henryRDlab/ElectricityTheftDetection/master/"
SGCC_PARTS = ("data.z01", "data.z02", "data.zip")
LISBON = (38.72, -9.14)
LOCAL_TZ = "Europe/Lisbon"


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def download_weather(start: str = "2011-01-01", end: str = "2014-12-31") -> Path:
    """Hourly temperature and humidity for Lisbon from the Open-Meteo archive (no key)."""
    dest = paths.RAW / "weather_lisbon.csv"
    if dest.exists():
        return dest
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LISBON[0],
            "longitude": LISBON[1],
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,relative_humidity_2m",
            "timezone": LOCAL_TZ,
        },
        timeout=60,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h).rename(columns={"time": "ts"})
    df.to_csv(dest, index=False)
    return dest


def download_all() -> None:
    paths.ensure_dirs()
    _fetch(UCI_URL, paths.RAW / "uci321.zip")
    for part in SGCC_PARTS:
        _fetch(SGCC_BASE + part, paths.RAW / part)
    download_weather()


# ---------------------------------------------------------------------------
# UCI ElectricityLoadDiagrams20112014
# ---------------------------------------------------------------------------


def load_uci_raw(path: Path | None = None) -> pd.DataFrame:
    """15-minute load in kW, one column per client, naive local (Portuguese) time."""
    path = path or paths.RAW / "uci321.zip"
    with zipfile.ZipFile(path) as z:
        with z.open("LD2011_2014.txt") as f:
            df = pd.read_csv(f, sep=";", decimal=",", index_col=0).astype(np.float32)
    df.index = pd.to_datetime(df.index)
    df.index.name = "ts"
    return df


def dst_transition_hours(index: pd.DatetimeIndex, tz: str = LOCAL_TZ) -> pd.DatetimeIndex:
    """Naive local 01:00 hour starts on clock-change days (23 h and 25 h days).

    The UCI documentation says the 01:00-02:00 values are zero (spring) or the aggregate of
    two hours (autumn) on those days, so the hour is unreliable.
    """
    return pd.DatetimeIndex(dst_transitions(index, tz)["ts"])


def to_hourly(raw15: pd.DataFrame) -> pd.DataFrame:
    """Mean kW over each hour (numerically equal to kWh per hour)."""
    return raw15.resample("1h").mean()


def prepare_uci() -> None:
    paths.ensure_dirs()
    raw15 = load_uci_raw()
    raw15.to_parquet(paths.PROCESSED / "uci_15min.parquet")
    hourly = to_hourly(raw15)
    hourly.to_parquet(paths.PROCESSED / "uci_hourly.parquet")


def clean_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Blank the DST-affected hours and interpolate them."""
    bad = dst_transition_hours(hourly.index)
    out = hourly.copy()
    out.loc[out.index.isin(bad)] = np.nan
    return out.interpolate(limit=3, limit_area="inside")


def active_panel(hourly: pd.DataFrame, start: str, max_zero_frac: float = 0.01) -> list[str]:
    """Clients that are metering for (almost) the whole period from ``start``."""
    sub = hourly.loc[start:]
    return list(sub.columns[(sub <= 0).mean() <= max_zero_frac])


def load_weather() -> pd.DataFrame:
    df = pd.read_csv(paths.RAW / "weather_lisbon.csv", parse_dates=["ts"], index_col="ts")
    df = df[~df.index.duplicated()]
    return df.rename(columns={"temperature_2m": "temp", "relative_humidity_2m": "humidity"})


# ---------------------------------------------------------------------------
# SGCC theft dataset
# ---------------------------------------------------------------------------


def extract_sgcc() -> Path:
    csv = paths.RAW / "sgcc" / "data.csv"
    if csv.exists():
        return csv
    csv.parent.mkdir(parents=True, exist_ok=True)
    blob = b"".join((paths.RAW / p).read_bytes() for p in SGCC_PARTS)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extract("data.csv", csv.parent)
    return csv


def load_sgcc() -> tuple[pd.DataFrame, pd.Series]:
    """Daily kWh (rows: days, columns: meters, NaN = missing) and the 0/1 theft label."""
    cache = paths.PROCESSED / "sgcc_daily.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        raw = pd.read_csv(extract_sgcc())
        raw = raw.set_index("CONS_NO")
        flags = raw.pop("FLAG").astype(int)
        raw.columns = pd.to_datetime(raw.columns, format="%Y/%m/%d")
        raw = raw.sort_index(axis=1).astype(np.float32)
        df = raw.T
        df.columns = df.columns.astype(str)
        paths.PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        flags.rename("flag").to_csv(paths.PROCESSED / "sgcc_labels.csv")
    flags = pd.read_csv(paths.PROCESSED / "sgcc_labels.csv", index_col=0)["flag"]
    flags.index = flags.index.astype(str)
    return df, flags.loc[df.columns]
