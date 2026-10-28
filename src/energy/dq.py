"""Meter data-quality checks.

All checks take a wide frame (DatetimeIndex, one column per meter) and return plain
numbers or small frames, so they can be run on 15-minute, hourly or daily data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Lisbon"


def dst_transitions(index: pd.DatetimeIndex, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Clock-change days inside ``index``: naive local 01:00 and kind ('spring' or 'autumn')."""
    rows = []
    for y in range(index.min().year, index.max().year + 1):
        local = pd.date_range(f"{y}-03-01", f"{y}-11-01", freq="D", tz=tz)
        length = (local[1:] - local[:-1]) / pd.Timedelta(hours=1)
        for day, hours in zip(local[:-1].tz_localize(None), length, strict=True):
            if hours != 24:
                rows.append((day + pd.Timedelta(hours=1), "spring" if hours < 24 else "autumn"))
    out = pd.DataFrame(rows, columns=["ts", "kind"])
    return out[out["ts"].between(index.min(), index.max())].reset_index(drop=True)


def _run_stats(same: np.ndarray, min_run: int) -> tuple[np.ndarray, np.ndarray]:
    """Longest run and number of runs >= min_run of ``True`` per column of a 2-D bool array.

    ``same`` marks steps that repeat the previous one: n repeats = n+1 identical values.
    """
    run = np.zeros(same.shape[1], dtype=np.int64)
    longest = np.zeros_like(run)
    n_runs = np.zeros_like(run)
    for row in same:
        run = (run + 1) * row
        np.maximum(longest, run, out=longest)
        n_runs += run == min_run - 1
    return longest + 1, n_runs


def leading_zero_steps(df: pd.DataFrame) -> pd.Series:
    """Steps before the first non-zero reading: the meter was not yet commissioned."""
    active = df.fillna(0).to_numpy() != 0
    first = np.where(active.any(axis=0), active.argmax(axis=0), len(df))
    return pd.Series(first, index=df.columns, name="leading_zero_steps")


def find_gaps(df: pd.DataFrame, freq: str) -> dict:
    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    missing_ts = expected.difference(df.index.unique())
    nan_steps = df.isna().sum()
    return {
        "missing_timestamps": int(len(missing_ts)),
        "meters_with_nan": int((nan_steps > 0).sum()),
        "nan_steps_total": int(nan_steps.sum()),
        "nan_steps": nan_steps,
    }


def find_duplicates(df: pd.DataFrame) -> dict:
    dup = df.index.duplicated(keep=False)
    conflicting = 0
    if dup.any():
        g = df[dup].groupby(level=0).nunique(dropna=False)
        conflicting = int((g > 1).any(axis=1).sum())
    return {
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "conflicting_timestamps": conflicting,
    }


def find_negatives(df: pd.DataFrame) -> pd.Series:
    return (df < 0).sum().rename("negative_steps")


def find_stuck(df: pd.DataFrame, min_run: int) -> pd.DataFrame:
    """Runs of >= ``min_run`` identical non-zero readings (frozen register / estimated values)."""
    x = df.to_numpy()
    same = (x[1:] == x[:-1]) & (x[1:] != 0) & ~np.isnan(x[1:])
    longest, n_runs = _run_stats(same, min_run)
    return pd.DataFrame({"longest_stuck": longest, "stuck_runs": n_runs}, index=df.columns)


def find_zero_runs(df: pd.DataFrame, min_run: int) -> pd.DataFrame:
    """Runs of >= ``min_run`` zero readings after the meter first reported consumption."""
    x = df.to_numpy()
    lead = leading_zero_steps(df).to_numpy()
    zero = x == 0
    zero &= np.arange(len(x))[:, None] >= lead[None, :]
    same = zero[1:] & zero[:-1]
    longest, n_runs = _run_stats(same, min_run)
    longest = np.where(zero.any(axis=0), longest, 0)
    return pd.DataFrame({"longest_zero_run": longest, "zero_runs": n_runs}, index=df.columns)


def dst_anomalies(df: pd.DataFrame, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Compare the 01:00-02:00 block on clock-change days with the neighbouring hours.

    ``zero_share``: share of active meters reading exactly zero in the block (spring: the hour
    does not exist). ``median_ratio``: median over meters of block mean / mean of the
    neighbouring hours (autumn: two hours folded into one gives a ratio near 2).
    """
    rows = []
    hour = pd.Timedelta(hours=1)
    step = pd.Timedelta(df.index.to_series().diff().median())
    active = df.loc[:, df.abs().sum() > 0]
    for _, r in dst_transitions(df.index, tz).iterrows():
        t = r["ts"]
        block = active.loc[t : t + hour - step]
        around = pd.concat(
            [active.loc[t - hour : t - step], active.loc[t + hour : t + 2 * hour - step]]
        )
        if block.empty or around.empty:
            continue
        b, a = block.mean(), around.mean()
        ok = a > 0
        rows.append(
            {
                "ts": t,
                "kind": r["kind"],
                "zero_share": float((block.max() == 0)[ok].mean()),
                "median_ratio": float((b[ok] / a[ok]).median()),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class DQConfig:
    freq: str
    stuck_min_run: int
    zero_min_run: int
    tz: str = LOCAL_TZ


def quality_report(df: pd.DataFrame, cfg: DQConfig) -> dict:
    """Run every check and return summary numbers plus a per-meter table."""
    gaps = find_gaps(df, cfg.freq)
    dups = find_duplicates(df)
    neg = find_negatives(df)
    stuck = find_stuck(df, cfg.stuck_min_run)
    zeros = find_zero_runs(df, cfg.zero_min_run)
    lead = leading_zero_steps(df)
    per_meter = pd.concat([lead, gaps["nan_steps"].rename("nan_steps"), neg, stuck, zeros], axis=1)
    summary = {
        "meters": int(df.shape[1]),
        "steps": int(df.shape[0]),
        "period": f"{df.index.min()} to {df.index.max()}",
        "freq": cfg.freq,
        "missing_timestamps": gaps["missing_timestamps"],
        "duplicate_timestamps": dups["duplicate_timestamps"],
        "conflicting_duplicates": dups["conflicting_timestamps"],
        "meters_with_nan": gaps["meters_with_nan"],
        "meters_with_negatives": int((neg > 0).sum()),
        "meters_with_leading_zeros": int((lead > 0).sum()),
        "meters_never_active": int((lead == len(df)).sum()),
        "meters_with_stuck_runs": int((stuck["stuck_runs"] > 0).sum()),
        "meters_with_zero_runs": int((zeros["zero_runs"] > 0).sum()),
        "stuck_min_run": cfg.stuck_min_run,
        "zero_min_run": cfg.zero_min_run,
    }
    return {"summary": summary, "per_meter": per_meter, "dst": dst_anomalies(df, cfg.tz)}


def to_markdown(report: dict, title: str = "Meter data-quality report") -> str:
    s = report["summary"]
    lines = [f"# {title}", "", "| Check | Result |", "|---|---|"]
    labels = {
        "meters": "Meters",
        "steps": "Timestamps per meter",
        "period": "Period",
        "freq": "Frequency",
        "missing_timestamps": "Missing timestamps (calendar gaps)",
        "duplicate_timestamps": "Duplicate timestamps",
        "conflicting_duplicates": "Duplicates with conflicting values",
        "meters_with_nan": "Meters with NaN readings",
        "meters_with_negatives": "Meters with negative readings",
        "meters_with_leading_zeros": "Meters with leading zeros (not yet commissioned)",
        "meters_never_active": "Meters that never report non-zero",
        "meters_with_stuck_runs": f"Meters with stuck runs (>= {s['stuck_min_run']} equal)",
        "meters_with_zero_runs": f"Meters with zero runs after activation (>= {s['zero_min_run']})",
    }
    for k, lab in labels.items():
        lines.append(f"| {lab} | {s[k]} |")
    dst = report["dst"]
    if len(dst):
        lines += ["", "## Clock-change days (01:00-02:00 block vs neighbouring hours)", ""]
        lines += [
            "| Timestamp | Kind | Share of meters at exactly 0 | Median ratio to neighbours |",
            "|---|---|---|---|",
        ]
        for _, r in dst.iterrows():
            lines.append(
                f"| {r['ts']} | {r['kind']} | {r['zero_share']:.2f} | {r['median_ratio']:.2f} |"
            )
    pm = report["per_meter"]
    worst = pm.sort_values(["longest_stuck", "longest_zero_run"], ascending=False).head(10)
    lines += ["", "## Ten meters with the longest stuck / zero runs", ""]
    lines.append("| Meter | Longest stuck run | Longest zero run | NaN steps | Leading zeros |")
    lines.append("|---|---|---|---|---|")
    for m, r in worst.iterrows():
        lines.append(
            f"| {m} | {int(r['longest_stuck'])} | {int(r['longest_zero_run'])} | "
            f"{int(r['nan_steps'])} | {int(r['leading_zero_steps'])} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# System-level load series (TSO data): one measured series plus the TSO's own forecasts
# ---------------------------------------------------------------------------


def _episodes(flag: pd.Series, max_gap: int = 2) -> pd.DataFrame:
    """Group flagged timestamps that are at most ``max_gap`` steps apart."""
    pos = np.flatnonzero(flag.to_numpy())
    if len(pos) == 0:
        return pd.DataFrame(columns=["start", "end", "quarters"])
    cuts = np.flatnonzero(np.diff(pos) > max_gap + 1) + 1
    groups = np.split(pos, cuts)
    return pd.DataFrame(
        {
            "start": [flag.index[g[0]] for g in groups],
            "end": [flag.index[g[-1]] for g in groups],
            "quarters": [len(g) for g in groups],
        }
    )


def tso_load_report(
    q: pd.DataFrame, raw_rows: int, z_cut: float = 6.0, local_tz: str = "Europe/Brussels"
) -> dict:
    """Checks for a quarter-hourly measured-load series with the TSO's forecast columns.

    ``q`` has columns load, da_fc, da_p10, da_p90, latest_fc on a UTC index; ``raw_rows`` is the
    row count before de-duplication. Measured load is compared with the TSO's most-recent
    forecast, which tracks it closely, so a large departure points to a metering or upscaling
    artefact rather than to real demand.
    """
    measured = q["load"].loc[: q["load"].last_valid_index()]
    grid = pd.date_range(q.index.min(), q.index.max(), freq="15min")
    valid = measured.dropna()
    dev = (valid - q["latest_fc"].loc[valid.index]).dropna()
    sigma = 1.4826 * (dev - dev.median()).abs().median()
    z = (dev - dev.median()) / sigma
    eps = _episodes(z.abs() > z_cut)
    if len(eps):
        eps["max_dev_mw"] = [
            float(dev.loc[a:b].abs().max()) for a, b in zip(eps["start"], eps["end"], strict=True)
        ]
    runs = (measured != measured.shift()).cumsum()
    return {
        "rows_raw": int(raw_rows),
        "rows": int(len(q)),
        "period": f"{q.index.min():%Y-%m-%d %H:%M} to {q.index.max():%Y-%m-%d %H:%M} UTC",
        "last_measured": f"{measured.index.max():%Y-%m-%d %H:%M} UTC",
        "duplicate_timestamps": int(raw_rows - len(q)),
        "missing_timestamps": int(len(grid.difference(q.index))),
        "missing_measured": [f"{t:%Y-%m-%d %H:%M}" for t in measured.index[measured.isna()]],
        "non_positive": int((measured <= 0).sum()),
        "stuck_runs_ge_4": int((measured.groupby(runs).transform("size") >= 4).sum()),
        "deviation_sigma_mw": float(sigma),
        "episodes": eps.sort_values("max_dev_mw", ascending=False) if len(eps) else eps,
        "flagged_quarters": int((z.abs() > z_cut).sum()),
        "forecast_band_violations": int(
            ((q["da_p10"] > q["da_fc"]) | (q["da_p90"] < q["da_fc"])).sum()
        ),
        "day_ahead_below_p10": float((valid < q["da_p10"].loc[valid.index]).mean()),
        "day_ahead_above_p90": float((valid > q["da_p90"].loc[valid.index]).mean()),
    }


def tso_to_markdown(r: dict, title: str) -> str:
    miss = ", ".join(r["missing_measured"]) or "none"
    lines = [
        f"# {title}",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Rows (raw / after de-duplication) | {r['rows_raw']} / {r['rows']} |",
        f"| Period | {r['period']} |",
        f"| Last measured quarter-hour | {r['last_measured']} |",
        f"| Duplicate timestamps | {r['duplicate_timestamps']} |",
        f"| Missing timestamps on the 15-minute grid | {r['missing_timestamps']} |",
        f"| Quarter-hours with no measured load (UTC) | {len(r['missing_measured'])}: {miss} |",
        f"| Non-positive load readings | {r['non_positive']} |",
        f"| Readings inside runs of 4 or more identical values | {r['stuck_runs_ge_4']} |",
        f"| Day-ahead p10-p90 band violations (p10 above forecast or p90 below it) | "
        f"{r['forecast_band_violations']} |",
        f"| Measured load below / above Elia's day-ahead p10 / p90 | "
        f"{r['day_ahead_below_p10']:.1%} / {r['day_ahead_above_p90']:.1%} |",
        f"| Robust sigma of (measured - most recent forecast) | {r['deviation_sigma_mw']:.0f} MW |",
        f"| Quarter-hours more than 6 sigma from the most recent forecast | "
        f"{r['flagged_quarters']} |",
        "",
        "## Largest departures from Elia's most recent forecast",
        "",
        "Consecutive flagged quarter-hours (gaps of up to two) are grouped into one episode. "
        "The data alone cannot say whether an episode is a metering or upscaling artefact or a "
        "real demand event that the forecast missed.",
        "",
        "| Start (UTC) | End (UTC) | Quarter-hours | Largest departure (MW) |",
        "|---|---|---|---|",
    ]
    for _, e in r["episodes"].head(8).iterrows():
        lines.append(
            f"| {e['start']:%Y-%m-%d %H:%M} | {e['end']:%Y-%m-%d %H:%M} | {int(e['quarters'])} | "
            f"{e['max_dev_mw']:.0f} |"
        )
    return "\n".join(lines) + "\n"
