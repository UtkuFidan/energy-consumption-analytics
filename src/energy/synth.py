"""Small synthetic generators used by the offline tests and the CI run."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hourly_load(days: int = 400, start: str = "2013-01-01", seed: int = 0) -> pd.Series:
    """Aggregate load with daily, weekly and annual cycles plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    hour = idx.hour.to_numpy()
    daily = 1 + 0.4 * np.sin(2 * np.pi * (hour - 9) / 24)
    weekly = np.where(idx.day_of_week >= 5, 0.8, 1.0)
    annual = 1 + 0.2 * np.cos(2 * np.pi * idx.day_of_year.to_numpy() / 365.25)
    y = 1000 * daily * weekly * annual * (1 + 0.03 * rng.standard_normal(len(idx)))
    return pd.Series(y, index=idx, name="load_kw")


def weather(index: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = (
        15
        - 8 * np.cos(2 * np.pi * index.day_of_year.to_numpy() / 365.25)
        + rng.normal(0, 1.5, len(index))
    )
    return pd.DataFrame({"temp": t, "humidity": 60.0}, index=index)


def daily_meters(
    n: int = 120, days: int = 400, start: str = "2014-01-01", seed: int = 0
) -> pd.DataFrame:
    """Daily kWh for ``n`` clean meters: level, weekly shape, annual cycle, noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="D")
    level = rng.lognormal(3.0, 0.5, n)
    dow = 1 + 0.25 * np.sin(
        2 * np.pi * (idx.day_of_week.to_numpy()[:, None] + rng.uniform(0, 7, n)) / 7
    )
    annual = 1 + 0.2 * np.cos(2 * np.pi * idx.day_of_year.to_numpy() / 365.25)[:, None]
    noise = 1 + 0.12 * rng.standard_normal((days, n))
    X = level[None, :] * dow * annual * noise
    return pd.DataFrame(np.clip(X, 0.05, None), index=idx, columns=[f"M{i:03d}" for i in range(n)])
