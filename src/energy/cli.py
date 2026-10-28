"""``energy`` command line interface."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import (
    data,
    dq,
    evaluation,
    experiments,
    forecast,
    markets,
    paths,
    plots,
    profiling,
    theft,
    tune,
)

PANEL_START = "2012-01-01"
UCI_BURN_IN, UCI_TEST_START, UCI_TEST_END = "2013-11-01", "2014-01-01", "2015-01-01"
BE_TZ = "Europe/Brussels"
POINT_MODELS = ["naive_day", "naive_week", "mstl", "lgbm_noweather", "lgbm", "xgb", "ens"]


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = [df.index.name or "", *map(str, df.columns)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.astype(object).iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float | np.floating) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines) + "\n"


def _write(name: str, text: str) -> None:
    (paths.RESULTS / name).write_text(text)


def _ci(v: float, lo: float, hi: float, nd: int = 0) -> str:
    return f"{v:,.{nd}f} ({lo:,.{nd}f} to {hi:,.{nd}f})"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def cmd_download(_a) -> None:
    data.download_all()
    markets.download_elia()
    markets.download_weather_forecast(markets.BE_POINTS, "weather_be.csv")
    markets.download_weather_forecast(
        markets.BE_POINTS, "weather_be_prev.csv", start="2024-01-01", previous_runs=True
    )
    markets.download_energinet("DK1")
    markets.download_weather_forecast(markets.DK_POINTS, "weather_dk.csv")
    markets.download_weather_forecast(
        markets.DK_POINTS, "weather_dk_prev.csv", start="2024-01-01", previous_runs=True
    )
    print("downloaded to", paths.RAW)


def cmd_prepare(_a) -> None:
    data.prepare_uci()
    X, y = data.load_sgcc()
    print(
        f"UCI hourly + 15-min cached; SGCC {X.shape[1]} meters x {X.shape[0]} days, "
        f"theft rate {y.mean():.4f}"
    )


def cmd_dq(a) -> None:
    paths.ensure_dirs()
    if a.dataset == "elia":
        raw_rows = len(pd.read_csv(paths.RAW / "elia_load.csv", sep=";", usecols=["datetime"]))
        q = markets.parse_elia_csv(paths.RAW / "elia_load.csv")
        rep = dq.tso_load_report(q, raw_rows)
        text = dq.tso_to_markdown(rep, "Data-quality report: Elia total load, 15-minute, 2023-2026")
        (paths.DOCS / "dq_elia.md").write_text(text)
        print(text)
        return
    raw = pd.read_parquet(paths.PROCESSED / "uci_15min.parquet")
    cfg = dq.DQConfig(
        freq="15min", stuck_min_run=96, zero_min_run=96
    )  # one day of identical / zero readings
    rep = dq.quality_report(raw, cfg)
    text = dq.to_markdown(
        rep, "Meter data-quality report: UCI ElectricityLoadDiagrams20112014 (15-minute raw)"
    )
    (paths.DOCS / "dq_report.md").write_text(text)
    rep["per_meter"].to_csv(paths.RESULTS / "dq_per_meter.csv")
    print(text)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def cmd_tune(a) -> None:
    paths.ensure_dirs()
    t0 = time.time()
    h, w = experiments.load_elia()
    feats = forecast.build_features(h["load"], w, forecast.BELGIUM).dropna(subset=["lag_2w"])
    cols = forecast.feature_columns(feats)
    folds = tune.validation_folds(feats.index, experiments.TUNE_MONTHS, forecast.BELGIUM)
    out_file = paths.RESULTS / f"forecast_elia_tuning_{a.model}.json"
    out = tune.tune(a.model, feats, cols, folds, a.trials)
    out |= {"validation_months": experiments.TUNE_MONTHS, "features": len(cols)}
    out_file.write_text(json.dumps(out, indent=2))
    print(a.model, json.dumps(out, indent=1), f"{time.time() - t0:.0f}s")


def _tuned() -> tuple[dict | None, dict | None]:
    found = [paths.RESULTS / f"forecast_elia_tuning_{m}.json" for m in ("lgbm", "xgb")]
    if not all(f.exists() for f in found):
        return None, None
    return tuple(json.loads(f.read_text())["params"] for f in found)


def _score_and_write(
    prefix: str, pred: pd.DataFrame, models: list[str], ref: str | None, naive: str, tz: str | None
) -> tuple[pd.DataFrame, evaluation.DayBootstrap]:
    tab, boot = experiments.score_table(pred, models, ref, naive, tz)
    tab.to_csv(paths.RESULTS / f"{prefix}_metrics.csv")
    pretty = pd.DataFrame(index=tab.index)
    pretty["MAE"] = [_ci(r.MAE, r.MAE_lo, r.MAE_hi) for r in tab.itertuples()]
    pretty["RMSE"] = tab["RMSE"].map("{:,.0f}".format)
    pretty["MAPE %"] = tab["MAPE_%"].map("{:.2f}".format)
    pretty[f"Skill vs {naive}"] = [
        _ci(r.skill_vs_naive, r.skill_vs_naive_lo, r.skill_vs_naive_hi, 3) for r in tab.itertuples()
    ]
    if ref:
        pretty[f"Skill vs {ref}"] = [
            _ci(r.skill_vs_ref, r.skill_vs_ref_lo, r.skill_vs_ref_hi, 3) for r in tab.itertuples()
        ]
        pretty["Days better"] = tab["days_better_than_ref"].map("{:.0%}".format)
    pretty.index.name = "model"
    _write(f"{prefix}_metrics.md", _md(pretty))
    return tab, boot


def _intervals_and_months(prefix: str, pred: pd.DataFrame, models: list[str], elia: bool,
                          tz: str | None) -> dict:  # fmt: skip
    iv = experiments.interval_table(pred, elia)
    iv.index.name = "interval"
    iv.to_csv(paths.RESULTS / f"{prefix}_intervals.csv")
    _write(f"{prefix}_intervals.md", _md(iv, "{:,.3f}"))
    months = experiments.mae_by_month(pred, models, tz)
    months.to_csv(paths.RESULTS / f"{prefix}_mae_by_month.csv")
    cov = {
        "raw": experiments.coverage_by_month(pred, "q10", "q90", tz),
        "conformalised": experiments.coverage_by_month(pred, "q10_cp", "q90_cp", tz),
    }
    return {
        "interval": iv,
        "months": months,
        "coverage_range": {k: (round(float(v.min()), 3), round(float(v.max()), 3))
                           for k, v in cov.items()},
    }  # fmt: skip


def _forecast_elia() -> None:
    t0 = time.time()
    h, w = experiments.load_elia()
    lgb_p, xgb_p = _tuned()
    pred = experiments.run_backtest(h["load"], w, forecast.BELGIUM, lgb_p, xgb_p, stack=h["da_fc"])
    for col, src in {"elia": "da_fc", "elia_p10": "da_p10", "elia_p90": "da_p90"}.items():
        pred[col] = h[src].reindex(pred.index)
    pred.to_parquet(paths.PROCESSED / "forecast_backtest_elia.parquet")
    models = [*POINT_MODELS, "lgbm_stack", "elia"]
    rep = experiments.clean_for_scoring(pred, [*models, "elia_p10", "elia_p90"],
                                        experiments.REPORT_START, BE_TZ)  # fmt: skip
    tab, boot = _score_and_write("forecast_elia", rep, models, "elia", "naive_week", BE_TZ)
    extra = _intervals_and_months("forecast_elia", rep, ["naive_week", "ens", "elia"], True, BE_TZ)
    hours = evaluation.error_by_hour(rep, ["naive_week", "ens", "elia"], BE_TZ)
    hours.to_csv(paths.RESULTS / "forecast_elia_mae_by_hour.csv")
    plots.forecast_vs_actual(
        pred.dropna(subset=["actual", "elia"]), "2026-03-30", "2026-04-13",
        paths.DOCS / "forecast_vs_actual.png",
        {"ens": (plots.BLUE, "Ours (GBM ensemble)"), "elia": (plots.VERMILION, "Elia day-ahead"),
         "naive_week": (plots.ORANGE, "Same hour last week")},
        "Day-ahead forecast vs measured load, Belgium (two weeks around Easter 2026)",
        "Hourly mean load (MW)", BE_TZ,
    )  # fmt: skip
    styles = {"naive_week": (plots.GREY, "Same hour last week"),
              "ens": (plots.BLUE, "Ours (GBM ensemble)"),
              "elia": (plots.VERMILION, "Elia day-ahead")}  # fmt: skip
    plots.error_by_hour(hours, paths.DOCS / "error_by_hour.png", styles,
                        "MAE by hour of day, Oct 2025 to Sep 2026", "MAE (MW)")  # fmt: skip
    plots.mae_by_month(extra["months"], paths.DOCS / "mae_by_month.png", styles,
                       "MAE by month", "MAE (MW)")  # fmt: skip
    window = pred[pred.index >= pd.Timestamp(experiments.REPORT_START, tz=BE_TZ)]
    print(f"scored hours: {len(rep)} of {len(window)}")
    print(tab.round(3).T.to_string())
    print(extra["interval"].round(3).to_string())
    print("monthly coverage ranges:", extra["coverage_range"])
    print(extra["months"].round(0).to_string())
    print(f"{time.time() - t0:.0f}s")


def _forecast_dk() -> None:
    t0 = time.time()
    y, w = experiments.load_denmark()
    lgb_p, xgb_p = _tuned()  # transferred as-is, not re-tuned
    pred = experiments.run_backtest(y, w, forecast.DENMARK, lgb_p, xgb_p)
    pred.to_parquet(paths.PROCESSED / "forecast_backtest_dk.parquet")
    models = [m for m in POINT_MODELS if m in pred]
    tz = forecast.DENMARK.tz
    rep = experiments.clean_for_scoring(pred, models, experiments.REPORT_START, tz)
    tab, _ = _score_and_write("forecast_dk", rep, models, None, "naive_week", tz)
    extra = _intervals_and_months("forecast_dk", rep, ["naive_week", "ens"], False, tz)
    print(f"scored hours: {len(rep)}")
    print(tab.round(3).T.to_string())
    print(extra["interval"].round(3).to_string())
    print(f"{time.time() - t0:.0f}s")


def _aggregate_load() -> tuple[pd.Series, list[str]]:
    hourly = data.clean_hourly(pd.read_parquet(paths.PROCESSED / "uci_hourly.parquet"))
    panel = data.active_panel(hourly, PANEL_START)
    y = hourly.loc[PANEL_START:, panel].sum(axis=1).rename("load_kw")
    return y, panel


def _forecast_uci() -> None:
    t0 = time.time()
    y, panel = _aggregate_load()
    w = data.load_weather()
    pred = forecast.backtest(y, w, UCI_BURN_IN, UCI_TEST_END, forecast.PORTUGAL_UCI)
    pred = forecast.conformalize(pred, lag_days=1)
    models = ["naive_day", "naive_week", "lgbm_noweather", "lgbm", "xgb", "ens"]
    rep = pred[pred.index >= pd.Timestamp(UCI_TEST_START)]
    tab, _ = _score_and_write("forecast_uci", rep, models, None, "naive_day", None)
    extra = _intervals_and_months("forecast_uci", rep, ["naive_day", "ens"], False, None)
    print(f"UCI panel: {len(panel)} clients, {len(y)} hours, {len(rep)} test hours")
    print(tab.round(3).T.to_string())
    print(extra["interval"].round(3).to_string())
    print(f"{time.time() - t0:.0f}s")


def cmd_forecast(a) -> None:
    paths.ensure_dirs()
    {"elia": _forecast_elia, "dk": _forecast_dk, "uci": _forecast_uci}[a.market]()


# ---------------------------------------------------------------------------
# Profiling and theft
# ---------------------------------------------------------------------------


def cmd_profile(_a) -> None:
    paths.ensure_dirs()
    hourly = data.clean_hourly(pd.read_parquet(paths.PROCESSED / "uci_hourly.parquet"))
    win = hourly.loc["2013-01-01":"2014-12-31"]
    prof = profiling.client_profiles(win)
    arch = profiling.fit_archetypes(prof, mean_kw=win.mean())
    arch.selection.to_csv(paths.RESULTS / "profile_k_selection.csv", index=False)
    arch.summary.to_csv(paths.RESULTS / "profile_archetypes.csv")
    arch.labels.to_csv(paths.RESULTS / "profile_labels.csv")
    _write(
        "profile_archetypes.md",
        _md(arch.summary.assign(clients=arch.summary["clients"].astype(int))),
    )
    plots.archetypes(arch.centroids, arch.summary["share_%"], paths.DOCS / "cluster_archetypes.png")
    plots.k_selection(arch.selection, arch.k, paths.DOCS / "k_selection.png")
    print(
        f"{len(prof)} of {win.shape[1]} clients profiled; k={arch.k}; "
        f"ARI(kmeans, ward)={arch.ari_vs_ward:.3f}"
    )
    print(arch.selection.round(3).to_string(index=False))
    print(arch.summary.round(2).to_string())


def _budgets(n: int, a) -> dict[str, int]:
    if n < 2000:  # synthetic UCI panel: a handful of meters
        return {"20": 20, "50": 50, "10%": n // 10}
    k = theft.inspection_budget(a.inspectors, a.per_day, a.days)
    return {"100": 100, f"budget({k})": k, "1000": 1000, "10%": n // 10}


def _theft_synthetic(a) -> None:
    hourly = data.clean_hourly(pd.read_parquet(paths.PROCESSED / "uci_hourly.parquet"))
    panel = data.active_panel(hourly, PANEL_START)
    daily = hourly.loc[PANEL_START:, panel].resample("1D").sum()
    daily = daily[daily.index < "2015-01-01"]
    X, kinds = theft.inject_theft(daily, frac=0.15)
    y = (kinds != "").astype(int).rename("flag")
    F = theft.compute_features(X)
    iso = theft.isolation_forest_scores(F)
    xg, imp = theft.xgb_oof_scores(F, y)
    ev = theft.evaluate(y, {"isolation_forest": iso, "xgboost": xg}, _budgets(len(y), a))
    ev.to_csv(paths.RESULTS / "theft_metrics_synthetic.csv")
    _write("theft_metrics_synthetic.md", _md(ev))
    imp.to_csv(paths.RESULTS / "theft_importance_synthetic.csv", header=["gain_importance"])
    k = int(y.sum())
    top, topx = (s.sort_values(ascending=False).head(k).index for s in (iso, xg))
    rec = pd.DataFrame(
        {
            "isolation_forest": [(kinds.loc[top] == p).sum() / (kinds == p).sum()
                                 for p in theft.PATTERNS],
            "xgboost": [(kinds.loc[topx] == p).sum() / (kinds == p).sum()
                        for p in theft.PATTERNS],
        },
        index=pd.Index(theft.PATTERNS, name="pattern"),
    )  # fmt: skip
    rec.to_csv(paths.RESULTS / "theft_recall_by_pattern_synthetic.csv")
    _write("theft_recall_by_pattern_synthetic.md", _md(rec))
    print(f"synthetic: {X.shape[1]} meters x {X.shape[0]} days, positives {k}")
    print(ev.round(3).to_string())
    print(f"recall by pattern at k={k}:\n", rec.round(3).to_string())


def _theft_sgcc(a) -> None:
    X, y = data.load_sgcc()
    F = theft.compute_features(X)
    F.to_parquet(paths.PROCESSED / "theft_features_sgcc.parquet")
    budgets = _budgets(len(y), a)
    budget_key = next(b for b in budgets if b.startswith("budget"))
    iso = theft.isolation_forest_scores(F)
    base, _ = theft.xgb_oof_scores(F, y, cols=theft.BASE_FEATURES)
    default, _ = theft.xgb_oof_scores(F, y)
    tuned, fold_ap, chosen, imp = theft.xgb_nested_oof(F, y, 5, a.trials)
    scores = {
        "isolation_forest": iso,
        "xgb_base_features": base,
        "xgb_all_features": default,
        "xgb_tuned": tuned,
    }
    ev = theft.bootstrap_metrics(y, scores, budgets)
    ev["lift@budget"] = ev[f"P@{budget_key}"] / float(y.mean())
    ev.to_csv(paths.RESULTS / "theft_metrics_sgcc.csv")
    imp.to_csv(paths.RESULTS / "theft_importance_sgcc.csv", header=["gain_importance"])
    (paths.RESULTS / "theft_tuning_sgcc.json").write_text(
        json.dumps({"outer_fold_pr_auc": fold_ap, "chosen_params": chosen}, indent=2)
    )
    pretty = pd.DataFrame(index=ev.index)
    pretty["PR-AUC (95% CI)"] = [
        _ci(r.PR_AUC, r.PR_AUC_lo, r.PR_AUC_hi, 3) for r in ev.itertuples()
    ]
    pretty["ROC-AUC"] = ev["ROC_AUC"].map("{:.3f}".format)
    for lab in budgets:
        col = f"P@{lab}"
        pretty[f"Precision@{lab}"] = [
            _ci(r[0], r[1], r[2], 3) for r in zip(ev[col], ev[f"{col}_lo"], ev[f"{col}_hi"],
                                                  strict=True)
        ]  # fmt: skip
    pretty["Lift@budget"] = ev["lift@budget"].map("{:.1f}x".format)
    pretty.index.name = "model"
    _write("theft_metrics_sgcc.md", _md(pretty))
    plots.pr_curve(
        y, {"xgboost": tuned, "isolation_forest": iso}, budgets[budget_key],
        paths.DOCS / "pr_curve.png",
    )  # fmt: skip
    thr = theft.fit_thresholds(F)
    pd.Series(thr, name="threshold").to_csv(paths.RESULTS / "theft_reason_thresholds.csv")
    lst = theft.build_inspection_list(F, tuned, thr, 200)
    lst.to_csv(paths.PROCESSED / "inspection_list_top200.csv")
    sample = lst.head(25).copy()
    sample["actual_label"] = y.loc[sample.index].to_numpy()
    sample.to_csv(paths.RESULTS / "inspection_list_sample.csv")
    codes = theft.reason_codes(F, thr)
    share = {c: float(codes.str.contains(c).mean()) for c in [*theft.RULES, "MODEL_SCORE_ONLY"]}
    top_codes = theft.reason_codes(F.loc[lst.index], thr)
    top_share = {
        c: float(top_codes.str.contains(c).mean()) for c in [*theft.RULES, "MODEL_SCORE_ONLY"]
    }
    (paths.RESULTS / "theft_reason_code_rates.json").write_text(
        json.dumps({"population": share, "top200": top_share}, indent=2)
    )
    print(
        f"sgcc: {X.shape[1]} meters x {X.shape[0]} days, positives {int(y.sum())} ({y.mean():.4f})"
    )
    print(pretty.to_string())
    print("outer-fold PR-AUC:", np.round(fold_ap, 3), "mean", round(float(np.mean(fold_ap)), 3))
    print(imp.round(3).head(10).to_string())
    print("top200 reason codes:", {k: round(v, 3) for k, v in top_share.items()})


def cmd_theft(a) -> None:
    paths.ensure_dirs()
    t0 = time.time()
    (_theft_synthetic if a.synthetic else _theft_sgcc)(a)
    print(f"{time.time() - t0:.0f}s")


def cmd_all(a) -> None:
    a.market = "elia"
    for fn in (cmd_download, cmd_prepare, cmd_dq, cmd_forecast, cmd_profile, cmd_theft):
        fn(a)
    a.dataset = "elia"
    cmd_dq(a)
    for market in ("dk", "uci"):
        a.market = market
        cmd_forecast(a)
    a.synthetic = True
    cmd_theft(a)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="energy", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, hlp in [
        ("download", cmd_download, "download Elia, Energinet, SGCC, UCI and Open-Meteo data"),
        ("prepare", cmd_prepare, "build hourly/daily caches under data/processed"),
        ("dq", cmd_dq, "data-quality report (UCI meters or Elia system load)"),
        ("tune", cmd_tune, "Optuna search for the Elia day-ahead GBMs"),
        ("forecast", cmd_forecast, "rolling-origin day-ahead backtest"),
        ("profile", cmd_profile, "customer load archetypes"),
        ("theft", cmd_theft, "theft / NTL detection"),
        ("all", cmd_all, "run everything"),
    ]:
        sp = sub.add_parser(name, help=hlp)
        sp.set_defaults(fn=fn, synthetic=False, dataset="uci", market="elia")
        if name == "dq":
            sp.add_argument("--dataset", choices=["uci", "elia"], default="uci")
        if name == "forecast":
            sp.add_argument("--market", choices=["elia", "dk", "uci"], default="elia")
        if name == "tune":
            sp.add_argument("--model", choices=["lgbm", "xgb"], required=True)
        if name in ("tune", "theft", "all"):
            sp.add_argument(
                "--trials", type=int, default=40 if name == "tune" else 20,
                help="Optuna trials (per model, or per outer fold for theft)",
            )  # fmt: skip
        if name in ("theft", "all"):
            sp.add_argument("--inspectors", type=int, default=10)
            sp.add_argument("--per-day", type=int, default=10, help="meters per inspector-day")
            sp.add_argument("--days", type=int, default=5)
        if name == "theft":
            sp.add_argument(
                "--synthetic", action="store_true", help="inject patterns into UCI clients"
            )
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
