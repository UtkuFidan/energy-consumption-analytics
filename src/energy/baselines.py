"""Statistical baseline: MSTL (24 h and 168 h seasonality) with a damped-trend ETS on the trend."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import MSTL

from .forecast import InfoSet, horizon_hours

PERIODS = (24, 168)


def _seasonal_ahead(component: np.ndarray, period: int, steps: np.ndarray) -> np.ndarray:
    """Value of a periodic component ``steps`` hours after its last observation."""
    n = len(component)
    return component[n - period * np.ceil(steps / period).astype(int) + steps - 1]


def mstl_day_ahead(
    y: pd.Series, days: pd.DatetimeIndex, info: InfoSet, window_days: int = 56
) -> pd.Series:
    """One forecast per local day in ``days``, each using only data up to that day's cut-off.

    The last ``window_days`` of observations are decomposed with MSTL; the two seasonal
    components are repeated forward and the trend is extended with a damped additive ETS.
    """
    y = y.asfreq("h").interpolate(limit=6, limit_area="inside")
    out = []
    for d in days:
        day0 = pd.Timestamp(d)
        start, stop = (
            (day0.tz_localize(info.tz).tz_convert("UTC"),
             (day0 + pd.Timedelta(days=1)).tz_localize(info.tz).tz_convert("UTC"))
            if info.tz else (day0, day0 + pd.Timedelta(days=1))
        )  # fmt: skip
        tgt = pd.date_range(start, stop, freq="h", inclusive="left")  # 23, 24 or 25 hours
        lead = horizon_hours(tgt, info)
        last = tgt[0] - pd.Timedelta(hours=int(lead[0]))  # newest observed hour
        hist = y.loc[last - pd.Timedelta(hours=24 * window_days - 1) : last]
        if hist.isna().any() or len(hist) < 24 * window_days:
            out.append(pd.Series(np.nan, index=tgt))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MSTL(hist.to_numpy(), periods=PERIODS).fit()
            trend = ETSModel(res.trend, error="add", trend="add", damped_trend=True).fit(disp=False)
            tf = trend.forecast(int(lead.max()))
        seas = sum(_seasonal_ahead(res.seasonal[:, i], p, lead) for i, p in enumerate(PERIODS))
        out.append(pd.Series(tf[lead - 1] + seas, index=tgt))
    return pd.concat(out).rename("mstl")
