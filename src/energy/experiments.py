"""Backtest runs for the three markets and the tables behind the README."""

from __future__ import annotations

import pandas as pd

from . import baselines, markets
from . import evaluation as ev
from . import forecast as fc

BE_POINTS = ["bru", "ant", "lie"]
DK_POINTS = ["cph", "aar"]
BURN_IN = "2025-07-01"  # calibration history for the conformal step, not scored
REPORT_START, REPORT_END = "2025-10-01", "2026-10-01"
TUNE_MONTHS = ["2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01", "2025-06-01"]


def load_elia() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly Elia frame (load and Elia's own forecasts) and the Belgian weather inputs."""
    q = markets.parse_elia_csv(markets.paths.RAW / "elia_load.csv")
    h = markets.elia_hourly(q)
    w = markets.load_weather_forecast(
        "weather_be.csv", BE_POINTS, fc.BELGIUM, "weather_be_prev.csv"
    )
    return h, w


def load_denmark() -> tuple[pd.Series, pd.DataFrame]:
    y = markets.parse_energinet(markets.paths.RAW / "energinet_dk1.json").asfreq("h")
    w = markets.load_weather_forecast(
        "weather_dk.csv", DK_POINTS, fc.DENMARK, "weather_dk_prev.csv"
    )
    return y, w


def local_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D", inclusive="left")


def run_backtest(
    y: pd.Series,
    w: pd.DataFrame | None,
    info: fc.InfoSet,
    lgb_params: dict | None,
    xgb_params: dict | None,
    start: str = BURN_IN,
    end: str = REPORT_END,
    report_start: str = REPORT_START,
    stack: pd.Series | None = None,
    with_mstl: bool = True,
) -> pd.DataFrame:
    """Backtest plus statistical baseline plus conformal intervals over [start, end)."""
    pred = fc.backtest(y, w, start, end, info, lgb_params, xgb_params, stack=stack)
    if with_mstl:
        m = baselines.mstl_day_ahead(y, local_days(report_start, end), info)
        pred["mstl"] = m.reindex(pred.index)
    lag = 2 if info.cutoff_h > 1 else 1
    return fc.conformalize(pred, tz=info.tz, lag_days=lag)


def score_table(
    pred: pd.DataFrame, models: list[str], ref: str | None, naive: str, tz: str | None
) -> tuple[pd.DataFrame, ev.DayBootstrap]:
    """Point metrics with day-block bootstrap CIs, skill vs the naive baseline and vs ``ref``."""
    boot = ev.DayBootstrap(pred, models, tz)
    rows = []
    for m in models:
        pt, ci = boot.point(m), boot.ci(m)
        r = {"model": m, **pt, "MAE_lo": ci["MAE"][0], "MAE_hi": ci["MAE"][1]}
        s, (lo, hi) = boot.skill_ci(m, naive)
        r.update({"skill_vs_naive": s, "skill_vs_naive_lo": lo, "skill_vs_naive_hi": hi})
        if ref:
            s, (lo, hi) = boot.skill_ci(m, ref)
            r.update({"skill_vs_ref": s, "skill_vs_ref_lo": lo, "skill_vs_ref_hi": hi,
                      "days_better_than_ref": boot.share_better(m, ref)})  # fmt: skip
        rows.append(r)
    return pd.DataFrame(rows).set_index("model"), boot


def interval_table(pred: pd.DataFrame, elia: bool) -> pd.DataFrame:
    a = pred["actual"]
    rows = {
        "raw quantile regression": ev.interval_report(a, pred["q10"], pred["q90"], pred["q50"]),
        "conformalised": ev.interval_report(a, pred["q10_cp"], pred["q90_cp"], pred["q50"]),
    }
    if elia:
        rows["Elia published"] = ev.interval_report(
            a, pred["elia_p10"], pred["elia_p90"], pred["elia"]
        )
    return pd.DataFrame(rows).T


def coverage_by_month(pred: pd.DataFrame, lo: str, hi: str, tz: str | None) -> pd.Series:
    inside = (pred["actual"] >= pred[lo]) & (pred["actual"] <= pred[hi])
    idx = pred.index.tz_convert(tz) if tz else pred.index
    return inside.groupby(idx.tz_localize(None).to_period("M") if tz else idx.to_period("M")).mean()


def mae_by_month(pred: pd.DataFrame, models: list[str], tz: str | None) -> pd.DataFrame:
    idx = pred.index.tz_convert(tz) if tz else pred.index
    per = idx.tz_localize(None).to_period("M") if tz else idx.to_period("M")
    err = pd.DataFrame({m: (pred[m] - pred["actual"]).abs() for m in models})
    out = err.groupby(per).mean()
    out.index = out.index.astype(str)
    out.index.name = "month"
    return out


def clean_for_scoring(pred: pd.DataFrame, models: list[str], start: str, tz: str | None):
    """Report window only, and only hours where every compared forecast and the actual exist."""
    s = pd.Timestamp(start, tz=tz) if tz else pd.Timestamp(start)
    p = pred[pred.index >= s]
    return p.dropna(subset=["actual", *models])
