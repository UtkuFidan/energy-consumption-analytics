"""Day-ahead hourly load forecasting: features, baselines, GBMs, quantiles, backtest.

Information set. The forecast for local day D is issued at gate closure, which for the Belgian
and Danish day-ahead markets is 12:00 on D-1. The newest load value the forecaster has is the
hour starting 11:00 on D-1, so ``InfoSet.cutoff_h = 13`` hours before local midnight of D.
Every load-derived feature of a target hour is read at or before that cut-off: lags are the
smallest whole number of days that clears it (24 h for the early hours of D, 48 h for the
rest), rolling statistics are evaluated at the cut-off, and the horizon is a feature. Weather
and calendar inputs are known for the whole of D. ``tests/test_forecast.py`` overwrites
everything after the cut-off and asserts that no feature moves.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 7
N_JOBS = -1
QUANTILES = (0.1, 0.5, 0.9)
WEATHER_COLS = ("temp", "temp_first", "rad", "cloud", "wind", "humidity")
WEATHER_DERIVED = ("temp_day_mean", "temp_prev_day_mean", "temp_roll72")


@dataclass(frozen=True)
class InfoSet:
    country: str
    tz: str | None  # time zone of the calendar; None when the index is already naive local time
    cutoff_h: int  # hours from the last observed hour start to local midnight of the target day


BELGIUM = InfoSet("BE", "Europe/Brussels", 13)  # gate closure 12:00 on D-1
DENMARK = InfoSet("DK", "Europe/Copenhagen", 13)
PORTUGAL_UCI = InfoSet("PT", None, 1)  # older benchmark: data up to 23:00 on D-1


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def calendar_features(index: pd.DatetimeIndex, country: str = "BE") -> pd.DataFrame:
    """Calendar fields from a naive local-time index."""
    years = range(index.min().year - 1, index.max().year + 2)
    hol = holidays.country_holidays(country, years=years)
    days = index.normalize()
    day_idx = pd.DatetimeIndex(days.unique())
    flag = pd.Series(day_idx.isin(pd.DatetimeIndex(list(hol.keys()))), index=day_idx)
    prev = flag.shift(1, fill_value=False)
    nxt = flag.shift(-1, fill_value=False)
    dow = pd.Series(day_idx.day_of_week, index=day_idx)
    # a working day squeezed between a holiday and the weekend (Friday after Ascension etc.)
    bridge = ((dow == 4) & prev) | ((dow == 0) & nxt)
    xmas = pd.Series(
        ((day_idx.month == 12) & (day_idx.day >= 24)) | ((day_idx.month == 1) & (day_idx.day <= 1)),
        index=day_idx,
    )
    doy = index.day_of_year.to_numpy()
    return pd.DataFrame(
        {
            "hour": index.hour,
            "dow": index.day_of_week,
            "month": index.month,
            "is_weekend": (index.day_of_week >= 5).astype(int),
            "is_holiday": flag.reindex(days).to_numpy().astype(int),
            "day_before_holiday": nxt.reindex(days).to_numpy().astype(int),
            "day_after_holiday": prev.reindex(days).to_numpy().astype(int),
            "is_bridge": bridge.reindex(days).to_numpy().astype(int),
            "christmas_period": xmas.reindex(days).to_numpy().astype(int),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )


def _local(index: pd.DatetimeIndex, info: InfoSet) -> pd.DatetimeIndex:
    return index.tz_convert(info.tz) if info.tz else index


def horizon_hours(index: pd.DatetimeIndex, info: InfoSet) -> np.ndarray:
    """Hours between the last observed hour start and each target hour (elapsed time)."""
    local = _local(index, info)
    since_midnight = (local - local.normalize()) / pd.Timedelta(hours=1)
    return (since_midnight + info.cutoff_h).to_numpy().astype(int)


def lag_features(y: pd.Series, info: InfoSet = BELGIUM) -> pd.DataFrame:
    """Load-derived features. Each one only reads y at or before the day-ahead cut-off.

    ``y`` must be a regular hourly series (gaps as NaN).
    """
    v = y.to_numpy(dtype=np.float64)
    n = len(v)
    pos = np.arange(n)
    lead = horizon_hours(y.index, info)
    k = 24 * np.ceil(lead / 24).astype(int)  # smallest whole-day lag that is already observed
    cut = pos - lead

    def at(p: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        ok = (p >= 0) & (p < n)
        out[ok] = v[p[ok]]
        return out

    roll = pd.Series(v)
    r24, s24, r168 = roll.rolling(24).mean(), roll.rolling(24).std(), roll.rolling(168).mean()

    def at_cut(s: pd.Series, shift: int = 0) -> np.ndarray:
        arr = s.to_numpy()
        out = np.full(n, np.nan)
        p = cut - shift
        ok = p >= 0
        out[ok] = arr[p[ok]]
        return out

    f = {
        "lag_d1": at(pos - k),
        "lag_d2": at(pos - k - 24),
        "lag_d3": at(pos - k - 48),
        "lag_w": at(pos - 168),
        "lag_2w": at(pos - 336),
        "same_hour_mean_7d": np.mean([at(pos - k - 24 * j) for j in range(7)], axis=0),
        "last_obs": at(cut),
        "roll_mean_24": at_cut(r24),
        "roll_std_24": at_cut(s24),
        "roll_mean_168": at_cut(r168),
        "horizon_h": lead.astype(float),
    }
    f["level_vs_week"] = f["roll_mean_24"] / at_cut(r24, 168)
    return pd.DataFrame(f, index=y.index)


def weather_features(w: pd.DataFrame, index: pd.DatetimeIndex, info: InfoSet) -> pd.DataFrame:
    """Weather for the target hour (a forecast, or a stand-in) and short-memory summaries."""
    w = w.reindex(index).interpolate(limit_direction="both")
    out = w[[c for c in WEATHER_COLS if c in w]].copy()
    day = _local(index, info).normalize()
    daily = w["temp"].groupby(day).transform("mean")
    out["temp_day_mean"] = daily
    out["temp_prev_day_mean"] = daily.shift(24)
    out["temp_roll72"] = w["temp"].rolling(72, min_periods=24).mean()
    return out


def build_features(
    y: pd.Series, weather: pd.DataFrame | None = None, info: InfoSet = BELGIUM
) -> pd.DataFrame:
    y = y.asfreq("h")
    local = _local(y.index, info)
    naive_local = local.tz_localize(None) if info.tz else local
    cal = calendar_features(naive_local, info.country)
    cal.index = y.index
    parts = [y.rename("y"), cal, lag_features(y, info)]
    if weather is not None:
        parts.append(weather_features(weather, y.index, info))
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Backtest splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    train_end: (
        pd.Timestamp
    )  # exclusive: newest target hour whose actual is known at the origin + 1h
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def rolling_origin_splits(
    index: pd.DatetimeIndex, test_start: str, test_end: str, info: InfoSet = PORTUGAL_UCI,
    freq: str = "MS",
) -> Iterator[Fold]:  # fmt: skip
    """Expanding-window folds on local calendar months.

    A fold's training targets stop at the gate-closure cut-off of its first test day, so a model
    is never fitted on an hour whose actual was not yet published at forecast time.
    """
    tz = index.tz
    starts = pd.date_range(test_start, test_end, freq=freq, inclusive="left", tz=info.tz or tz)
    bounds = list(starts) + [pd.Timestamp(test_end, tz=info.tz or tz)]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if a < index.min() or b > index.max() + pd.Timedelta(hours=1):
            continue
        yield Fold(a - pd.Timedelta(hours=info.cutoff_h - 1), a, b)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

LGB_DEFAULTS = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
    "subsample": 0.8, "colsample_bytree": 0.8, "target": "level",
}  # fmt: skip
XGB_DEFAULTS = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
    "colsample_bytree": 0.8, "target": "level",
}  # fmt: skip


def _split_target(params: dict) -> tuple[dict, str]:
    p = dict(params)
    return p, p.pop("target", "level")


def make_lgb(params: dict | None = None, **kw) -> lgb.LGBMRegressor:
    p, _ = _split_target({**LGB_DEFAULTS, **(params or {})})
    return lgb.LGBMRegressor(
        subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=N_JOBS, **{**p, **kw}
    )


def make_xgb(params: dict | None = None, **kw) -> xgb.XGBRegressor:
    p, _ = _split_target({**XGB_DEFAULTS, **(params or {})})
    return xgb.XGBRegressor(random_state=SEED, n_jobs=N_JOBS, tree_method="hist", **{**p, **kw})


def fit_predict(make, params: dict | None, tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                defaults: dict, **kw) -> np.ndarray:  # fmt: skip
    """Fit on ``tr`` and predict ``te``. ``target='diff_w'`` models y - lag_w instead of y."""
    target = {**defaults, **(params or {})}.get("target", "level")
    base_tr = tr["lag_w"] if target == "diff_w" else 0.0
    base_te = te["lag_w"] if target == "diff_w" else 0.0
    label = tr["y"] - base_tr
    keep = label.notna().to_numpy()  # a missing hour a week earlier leaves no residual to learn
    model = make(params, **kw)
    model.fit(tr.loc[keep, cols], label[keep])
    return model.predict(te[cols]) + base_te


def feature_columns(df: pd.DataFrame, weather: bool = True, stack: bool = False) -> list[str]:
    drop = {"y", "elia_da"}
    if stack:
        drop.discard("elia_da")
    if not weather:
        drop |= set(WEATHER_COLS) | set(WEATHER_DERIVED)
    return [c for c in df.columns if c not in drop]


def backtest(
    y: pd.Series,
    weather: pd.DataFrame | None,
    test_start: str,
    test_end: str,
    info: InfoSet = PORTUGAL_UCI,
    lgb_params: dict | None = None,
    xgb_params: dict | None = None,
    quantiles: tuple[float, ...] = QUANTILES,
    stack: pd.Series | None = None,
) -> pd.DataFrame:
    """Monthly rolling-origin backtest. One row per test hour with every forecast.

    ``stack`` optionally adds a second LightGBM that also sees a reference forecast (Elia's own
    day-ahead forecast) as a feature; it is reported separately and is not the headline model.
    """
    feats = build_features(y, weather, info)
    if stack is not None:
        feats["elia_da"] = stack.reindex(feats.index)
    feats = feats.dropna(subset=["lag_2w"])
    has_weather = weather is not None
    out = []
    for fold in rolling_origin_splits(feats.index, test_start, test_end, info):
        tr = feats[(feats.index < fold.train_end) & feats["y"].notna()]
        te = feats[(feats.index >= fold.test_start) & (feats.index < fold.test_end)]
        res = pd.DataFrame(
            {"actual": te["y"], "naive_day": te["lag_d1"], "naive_week": te["lag_w"]}
        )
        cols = feature_columns(feats, has_weather)
        res["lgbm"] = fit_predict(make_lgb, lgb_params, tr, te, cols, LGB_DEFAULTS)
        res["xgb"] = fit_predict(make_xgb, xgb_params, tr, te, cols, XGB_DEFAULTS)
        res["ens"] = (res["lgbm"] + res["xgb"]) / 2
        if has_weather:
            nw = feature_columns(feats, False)
            res["lgbm_noweather"] = fit_predict(make_lgb, lgb_params, tr, te, nw, LGB_DEFAULTS)
        if stack is not None:
            sc = feature_columns(feats, has_weather, stack=True)
            res["lgbm_stack"] = fit_predict(make_lgb, lgb_params, tr, te, sc, LGB_DEFAULTS)
        qs = np.column_stack(
            [
                fit_predict(make_lgb, lgb_params, tr, te, cols, LGB_DEFAULTS,
                            objective="quantile", alpha=q)
                for q in quantiles
            ]
        )  # fmt: skip
        qs = np.sort(qs, axis=1)  # no quantile crossing
        for i, q in enumerate(quantiles):
            res[f"q{round(q * 100):02d}"] = qs[:, i]
        res["fold"] = fold.test_start
        out.append(res)
    return pd.concat(out)


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


def conformalize(
    pred: pd.DataFrame, lo: str = "q10", hi: str = "q90", level: float = 0.8,
    window_days: int = 60, lag_days: int = 2, tz: str | None = None,
) -> pd.DataFrame:  # fmt: skip
    """Rolling conformalised quantile regression (CQR) on already-published forecasts.

    For local day D the interval is widened by the ``level`` quantile of the conformity scores
    max(lo - y, y - hi) of the ``window_days`` days that end ``lag_days`` before D. With gate
    closure at 12:00 on D-1, day D-1 is only partly observed, so ``lag_days=2`` is the honest
    value; the UCI benchmark (data up to 23:00 on D-1) uses 1. Days without a week of history
    are left unadjusted.
    """
    out = pred.copy()
    score = np.maximum(pred[lo] - pred["actual"], pred["actual"] - pred[hi]).to_numpy()
    idx = pred.index.tz_convert(tz) if tz else pred.index
    days = idx.normalize()
    adj = np.zeros(len(pred))
    for d in days.unique():
        hist = score[(days <= d - pd.Timedelta(days=lag_days))
                     & (days > d - pd.Timedelta(days=lag_days + window_days))]  # fmt: skip
        hist = hist[np.isfinite(hist)]
        if len(hist) >= 7 * 24:
            adj[days == d] = np.quantile(hist, level)
    out[f"{lo}_cp"] = pred[lo] - adj
    out[f"{hi}_cp"] = pred[hi] + adj
    return out
