"""System-level load data: Elia (Belgium), Energinet (Denmark) and Open-Meteo weather.

Downloaders cache raw files under ``data/raw``; the parsers are pure functions of the file
contents so the tests can run them on small fixtures without a network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import paths

ELIA_DATASET = "ods001"  # Measured and forecasted total load on the Belgian grid (historical)
ELIA_EXPORT = (
    f"https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{ELIA_DATASET}/exports/csv"
)
ELIA_FIELDS = (
    "datetime,resolutioncode,totalload,dayaheadforecast,dayaheadconfidence10,"
    "dayaheadconfidence90,weekaheadforecast,mostrecentforecast"
)
ELIA_COLUMNS = {
    "totalload": "load",
    "dayaheadforecast": "da_fc",
    "dayaheadconfidence10": "da_p10",
    "dayaheadconfidence90": "da_p90",
    "weekaheadforecast": "wa_fc",
    "mostrecentforecast": "latest_fc",
}
STUDY_START = "2023-01-01"
STUDY_END = "2026-09-30"  # last full month with measured load at the time of the run

OPEN_METEO_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_PREVIOUS = "https://previous-runs-api.open-meteo.com/v1/forecast"
PUBLICATION_LAG_H = 6  # allowance between a model run's nominal time and its availability
WEATHER_VARS = ("temperature_2m", "shortwave_radiation", "cloud_cover", "wind_speed_10m")
BE_POINTS = {"bru": (50.85, 4.35), "ant": (51.22, 4.40), "lie": (50.63, 5.57)}
DK_POINTS = {"cph": (55.68, 12.57), "aar": (56.16, 10.21)}

ENERGINET_URL = "https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement"


def _get(url: str, params: dict | None = None, timeout: int = 300) -> requests.Response:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------------
# Elia
# ---------------------------------------------------------------------------


def download_elia(start: str = STUDY_START, end: str = STUDY_END, force: bool = False) -> Path:
    """Quarter-hourly load and Elia's own forecasts via the Opendatasoft exports endpoint."""
    dest = paths.RAW / "elia_load.csv"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    stop = pd.Timestamp(end) + pd.Timedelta(days=1)
    where = f"datetime>=date'{start}' AND datetime<date'{stop:%Y-%m-%d}'"
    r = _get(
        ELIA_EXPORT,
        {"where": where, "order_by": "datetime", "delimiter": ";", "select": ELIA_FIELDS},
    )
    dest.write_bytes(r.content)
    return dest


def parse_elia_csv(src) -> pd.DataFrame:
    """Parse an Elia export (path or file-like) into a UTC quarter-hour frame.

    Columns: load, da_fc, da_p10, da_p90, wa_fc, latest_fc (all MW). Rows that are not
    quarter-hourly are dropped, exact duplicates removed, conflicting duplicates keep the last.
    """
    raw = pd.read_csv(src, sep=";", encoding="utf-8-sig")
    raw["ts"] = pd.to_datetime(raw["datetime"], utc=True)
    raw = raw[raw["resolutioncode"] == "PT15M"]
    df = raw.set_index("ts")[list(ELIA_COLUMNS)].rename(columns=ELIA_COLUMNS).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.astype("float64")


def elia_hourly(q: pd.DataFrame) -> pd.DataFrame:
    """Hourly means of the quarter-hours; an hour with a missing quarter is NaN."""
    g = q.resample("1h")
    out = g.mean()
    out[g.count() < 4] = np.nan
    return out


# ---------------------------------------------------------------------------
# Open-Meteo (historical forecast API: archived model forecasts, not reanalysis)
# ---------------------------------------------------------------------------


def parse_open_meteo(payload: list | dict, names: list[str]) -> pd.DataFrame:
    """Open-Meteo multi-point response -> UTC hourly frame with columns ``<var>_<point>``."""
    payload = payload if isinstance(payload, list) else [payload]
    frames = []
    for name, p in zip(names, payload, strict=True):
        h = pd.DataFrame(p["hourly"])
        h.index = pd.DatetimeIndex(pd.to_datetime(h.pop("time"))).tz_localize("UTC")
        frames.append(h.add_suffix(f"_{name}"))
    return pd.concat(frames, axis=1)


def _fetch_points(url: str, points: dict, hourly: list[str], start: str, end: str) -> pd.DataFrame:
    names = list(points)
    chunks = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        a = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01"))
        b = min(pd.Timestamp(end), pd.Timestamp(f"{year}-12-31"))
        params = {
            "latitude": ",".join(str(points[n][0]) for n in names),
            "longitude": ",".join(str(points[n][1]) for n in names),
            "start_date": f"{a:%Y-%m-%d}",
            "end_date": f"{b:%Y-%m-%d}",
            "hourly": ",".join(hourly),
            "timezone": "UTC",
        }
        for attempt in range(4):  # the free tier answers 429 under bursts
            try:
                chunks.append(parse_open_meteo(_get(url, params, timeout=120).json(), names))
                break
            except requests.HTTPError:
                if attempt == 3:
                    raise
                time.sleep(20 * (attempt + 1))
    return pd.concat(chunks).rename_axis("ts")


def download_weather_forecast(
    points: dict[str, tuple[float, float]],
    dest_name: str,
    start: str = STUDY_START,
    end: str = STUDY_END,
    force: bool = False,
    previous_runs: bool = False,
) -> Path:
    """Archived forecasts. ``previous_runs`` fetches the forecasts issued 1 and 2 days earlier."""
    dest = paths.RAW / dest_name
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if previous_runs:
        hourly = [f"{v}_previous_day{d}" for v in WEATHER_VARS for d in (1, 2)]
        df = _fetch_points(OPEN_METEO_PREVIOUS, points, hourly, start, end)
    else:
        df = _fetch_points(OPEN_METEO_FORECAST, points, list(WEATHER_VARS), start, end)
    df.to_csv(dest)
    return dest


def gate_weather(
    stitched: pd.DataFrame, previous: pd.DataFrame | None, lead_h: np.ndarray
) -> pd.DataFrame:
    """Weather columns as they were forecast before day-ahead gate closure.

    ``stitched`` (historical forecast API) strings together the first hours of successive model
    runs, so it is closer to the truth than a forecast issued the day before. Where the previous
    runs API has data, each hour takes the forecast issued one day earlier if that run was
    available before gate closure (``lead_h`` hours after the last observed hour start, minus a
    publication allowance) and the one issued two days earlier otherwise.
    """
    out = stitched.copy()
    if previous is None:
        return out
    prev = previous.reindex(stitched.index)
    use_d1 = lead_h <= 24 - PUBLICATION_LAG_H + 1
    for col in stitched.columns:
        var, point = col.rsplit("_", 1)
        d1, d2 = prev.get(f"{var}_previous_day1_{point}"), prev.get(f"{var}_previous_day2_{point}")
        if d1 is None or d2 is None:
            continue
        out[col] = np.where(use_d1, d1, d2)
        out[col] = out[col].fillna(stitched[col])
    return out


def load_weather_forecast(
    name: str, points: list[str], info, previous_name: str | None = None
) -> pd.DataFrame:
    from .forecast import horizon_hours  # local import: forecast does not depend on markets

    stitched = pd.read_csv(paths.RAW / name, index_col="ts", parse_dates=True)
    lead_h = horizon_hours(stitched.index, info)
    prev = None
    if previous_name and (paths.RAW / previous_name).exists():
        prev = pd.read_csv(paths.RAW / previous_name, index_col="ts", parse_dates=True)
    return weather_inputs(gate_weather(stitched, prev, lead_h), points)


def weather_inputs(w: pd.DataFrame, points: list[str]) -> pd.DataFrame:
    def mean(var: str) -> pd.Series:
        return w[[f"{var}_{p}" for p in points]].mean(axis=1)

    out = pd.DataFrame(
        {
            "temp": mean("temperature_2m"),
            "temp_first": w[f"temperature_2m_{points[0]}"],
            "rad": mean("shortwave_radiation"),
            "cloud": mean("cloud_cover"),
            "wind": mean("wind_speed_10m"),
        }
    )
    return out[~out.index.duplicated()]


# ---------------------------------------------------------------------------
# Energinet (Denmark)
# ---------------------------------------------------------------------------


def download_energinet(
    area: str = "DK1", start: str = STUDY_START, end: str = STUDY_END, force: bool = False
) -> Path:
    dest = paths.RAW / f"energinet_{area.lower()}.json"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = _get(
        ENERGINET_URL,
        {
            "start": f"{start}T00:00",
            "end": f"{pd.Timestamp(end) + pd.Timedelta(days=1):%Y-%m-%d}T00:00",
            "filter": json.dumps({"PriceArea": [area]}),
            "columns": "HourUTC,PriceArea,GrossConsumptionMWh",
            "sort": "HourUTC",
            "limit": 0,
        },
    )
    dest.write_bytes(r.content)
    return dest


def parse_energinet(src) -> pd.Series:
    """Hourly gross consumption (MWh per hour, i.e. mean MW) indexed by UTC hour start."""
    text = src.read() if hasattr(src, "read") else Path(src).read_text()
    rec = pd.DataFrame(json.loads(text)["records"])
    ts = pd.to_datetime(rec["HourUTC"]).dt.tz_localize("UTC")
    s = pd.Series(rec["GrossConsumptionMWh"].to_numpy(dtype="float64"), index=ts, name="load")
    return s[~s.index.duplicated(keep="last")].sort_index()
