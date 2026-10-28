import numpy as np
import pandas as pd
import pytest

from energy import evaluation, forecast, synth

BE = forecast.BELGIUM
UCI = forecast.PORTUGAL_UCI


@pytest.fixture(scope="module")
def y():
    return synth.hourly_load(days=60)


@pytest.fixture(scope="module")
def y_utc():
    """Belgian-style hourly series on a UTC index, spanning the March clock change."""
    idx = pd.date_range("2025-02-01", periods=24 * 90, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.Series(
        1000 + rng.normal(0, 50, len(idx)) + 10 * (np.arange(len(idx)) % 97), index=idx
    )


def local_day(index, day, tz="Europe/Brussels"):
    return index.tz_convert(tz).normalize() == pd.Timestamp(day, tz=tz)


# --- information set -------------------------------------------------------------------


def test_horizon_is_13_to_36_hours_with_gate_closure_at_noon(y_utc):
    lead = forecast.horizon_hours(y_utc.index, BE)
    assert lead.min() == 13
    local = y_utc.index.tz_convert("Europe/Brussels")
    normal_day = local_day(y_utc.index, "2025-02-10")
    assert lead[normal_day].tolist() == list(range(13, 37))
    spring = local_day(y_utc.index, "2025-03-30")  # 23-hour day
    assert lead[spring].max() == 35 and spring.sum() == 23
    assert (local[normal_day].hour == np.arange(24)).all()


def test_lags_use_the_newest_whole_day_that_is_already_observed(y_utc):
    f = forecast.lag_features(y_utc, BE)
    day = pd.Timestamp("2025-02-10", tz="Europe/Brussels")
    for hour in (0, 10, 11, 12, 23):
        t = (day + pd.Timedelta(hours=hour)).tz_convert("UTC")
        back = 24 if hour <= 11 else 48  # 11:00 on D-1 is the last observed hour start
        assert f.loc[t, "lag_d1"] == y_utc.loc[t - pd.Timedelta(hours=back)]
        assert f.loc[t, "lag_w"] == y_utc.loc[t - pd.Timedelta(hours=168)]
        cut = day - pd.Timedelta(hours=13)
        assert f.loc[t, "last_obs"] == y_utc.loc[cut]
        assert f.loc[t, "roll_mean_24"] == pytest.approx(
            y_utc.loc[cut - pd.Timedelta(hours=23) : cut].mean()
        )


@pytest.mark.parametrize("info,tz", [(BE, "UTC"), (UCI, None)])
def test_features_for_day_d_ignore_everything_after_the_cutoff(info, tz):
    """Overwrite every observation the forecaster would not have; no feature of D may move."""
    idx = pd.date_range("2025-02-01", periods=24 * 60, freq="h", tz=tz)
    rng = np.random.default_rng(1)
    y = pd.Series(1000 + rng.normal(0, 50, len(idx)), index=idx)
    w = pd.DataFrame({"temp": rng.normal(8, 3, len(idx)), "rad": 100.0}, index=idx)
    day = pd.Timestamp("2025-03-05", tz="Europe/Brussels" if tz else None)
    first_unseen = day - pd.Timedelta(hours=info.cutoff_h - 1)  # cut-off hour + 1
    clean = forecast.build_features(y, w, info)
    tampered = y.copy()
    tampered[tampered.index >= first_unseen] = 1e9
    dirty = forecast.build_features(tampered, w, info)
    local = clean.index.tz_convert(info.tz) if info.tz else clean.index
    rows = local.normalize() == day
    cols = [c for c in clean.columns if c != "y"]
    assert rows.sum() >= 23
    pd.testing.assert_frame_equal(clean.loc[rows, cols], dirty.loc[rows, cols])
    # and the guard has teeth: features of D do move if the last observed hour is changed
    tampered = y.copy()
    tampered[tampered.index == first_unseen - pd.Timedelta(hours=1)] = 1e9
    moved = forecast.build_features(tampered, w, info)
    assert not clean.loc[rows, cols].equals(moved.loc[rows, cols])


def test_training_targets_stop_at_the_cutoff_of_the_first_test_day():
    idx = pd.date_range("2025-01-01", "2025-06-30 23:00", freq="h", tz="UTC")
    folds = list(forecast.rolling_origin_splits(idx, "2025-03-01", "2025-05-01", BE))
    assert len(folds) == 2
    f = folds[0]
    assert f.test_start == pd.Timestamp("2025-03-01", tz="Europe/Brussels")
    # newest training target = 11:00 on the last day before the test month
    assert f.train_end - pd.Timedelta(hours=1) == pd.Timestamp(
        "2025-02-28 11:00", tz="Europe/Brussels"
    )
    assert folds[0].test_end == folds[1].test_start


def test_conformal_scores_come_from_days_that_are_fully_observed():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=24 * 100, freq="h", tz="UTC")
    pred = pd.DataFrame({"actual": rng.standard_normal(len(idx))}, index=idx)
    pred["q10"], pred["q50"], pred["q90"] = -0.1, 0.0, 0.1
    base = forecast.conformalize(pred, tz="UTC", lag_days=2)
    day = pd.Timestamp("2025-04-05", tz="UTC")
    tampered = pred.copy()
    tampered.loc[tampered.index >= day - pd.Timedelta(days=1), "actual"] = 1e6  # D-1 onward
    after = forecast.conformalize(tampered, tz="UTC", lag_days=2)
    on_day = (base.index >= day) & (base.index < day + pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(
        base.loc[on_day], after.loc[on_day].assign(actual=base.loc[on_day, "actual"])
    )


# --- features and models ---------------------------------------------------------------


def test_lag_features_are_exact_shifts_for_the_uci_information_set(y):
    f = forecast.lag_features(y, UCI)
    t = y.index[500]  # 20:00; data up to 23:00 on the previous day is known
    cut = t.normalize() - pd.Timedelta(hours=1)
    assert f.loc[t, "lag_d1"] == y.loc[t - pd.Timedelta(hours=24)]
    assert f.loc[t, "lag_w"] == y.loc[t - pd.Timedelta(hours=168)]
    assert f.loc[t, "last_obs"] == y.loc[cut]
    assert f.loc[t, "roll_mean_24"] == pytest.approx(
        y.loc[cut - pd.Timedelta(hours=23) : cut].mean()
    )


def test_holiday_flags_belgium():
    idx = pd.date_range("2025-07-20", periods=72, freq="h")
    cal = forecast.calendar_features(idx, "BE")
    by_day = cal.groupby(idx.normalize()).first()
    assert by_day.loc["2025-07-21", "is_holiday"] == 1  # Belgian National Day
    assert by_day.loc["2025-07-20", "day_before_holiday"] == 1
    assert by_day.loc["2025-07-22", "day_after_holiday"] == 1
    assert by_day.loc["2025-07-20", "is_holiday"] == 0


def test_bridge_day_after_ascension():
    idx = pd.date_range("2025-05-28", periods=72, freq="h")  # Ascension Thursday is 29 May 2025
    cal = forecast.calendar_features(idx, "BE")
    by_day = cal.groupby(idx.normalize()).first()
    assert by_day.loc["2025-05-29", "is_holiday"] == 1
    assert by_day.loc["2025-05-30", "is_bridge"] == 1


def test_rolling_origin_splits_are_ordered_and_disjoint():
    idx = pd.date_range("2012-01-01", "2014-12-31 23:00", freq="h")
    folds = list(forecast.rolling_origin_splits(idx, "2014-01-01", "2015-01-01", UCI))
    assert len(folds) == 12
    for f in folds:
        assert f.train_end <= f.test_start < f.test_end
    for a, b in zip(folds[:-1], folds[1:], strict=True):
        assert a.test_end == b.test_start  # contiguous, no overlap
        assert a.train_end < b.train_end  # expanding window moves forward


def test_metrics_and_skill():
    a = pd.Series([100.0, 200.0])
    p = pd.Series([110.0, 180.0])
    m = evaluation.metrics(a, p)
    assert m["MAE"] == 15.0
    assert m["RMSE"] == pytest.approx(np.sqrt(250))
    assert m["MAPE_%"] == pytest.approx(10.0)
    assert evaluation.skill(5.0, 10.0) == 0.5
    assert evaluation.pinball(a, p, 0.9) == pytest.approx(9.5)


def test_backtest_predicts_only_test_window_and_quantiles_are_ordered():
    y = synth.hourly_load(days=120)
    w = synth.weather(y.index)
    fast = {"n_estimators": 60}
    pred = forecast.backtest(y, w, "2013-04-01", "2013-05-01", UCI, fast, fast)
    assert pred.index.min() >= pd.Timestamp("2013-04-01")
    assert pred.index.max() < pd.Timestamp("2013-05-01")
    assert pred.index.is_monotonic_increasing
    assert np.isfinite(pred[["lgbm", "xgb", "ens", "naive_day", "naive_week"]].to_numpy()).all()
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()
    # a smooth synthetic series is easy: the GBM must beat yesterday-as-forecast
    assert (
        evaluation.metrics(pred["actual"], pred["lgbm"])["MAE"]
        < evaluation.metrics(pred["actual"], pred["naive_day"])["MAE"]
    )


def test_residual_target_adds_the_weekly_lag_back():
    y = synth.hourly_load(days=60)
    feats = forecast.build_features(y, None, UCI).dropna(subset=["lag_2w"])
    tr, te = feats.iloc[:-48], feats.iloc[-48:]
    cols = forecast.feature_columns(feats, False)
    p = {"n_estimators": 50, "target": "diff_w"}
    out = forecast.fit_predict(forecast.make_lgb, p, tr, te, cols, forecast.LGB_DEFAULTS)
    assert abs(out - te["y"]).mean() < abs(te["lag_w"] - te["y"]).mean() * 1.5


def test_conformalize_repairs_too_narrow_intervals():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2014-01-01", periods=24 * 120, freq="h")
    pred = pd.DataFrame({"actual": rng.standard_normal(len(idx))}, index=idx)
    pred["q10"], pred["q50"], pred["q90"] = -0.1, 0.0, 0.1  # far too narrow
    out = forecast.conformalize(pred, lag_days=1)
    a = out["actual"]
    raw = evaluation.interval_report(a, out["q10"], out["q90"], out["q50"])["coverage"]
    fixed = evaluation.interval_report(a, out["q10_cp"], out["q90_cp"], out["q50"])["coverage"]
    assert raw < 0.2 and 0.7 < fixed < 0.9


def test_day_bootstrap_ci_brackets_the_point_estimate_and_orders_models():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=24 * 90, freq="h")
    actual = pd.Series(100 + rng.normal(0, 5, len(idx)), index=idx)
    pred = pd.DataFrame({"actual": actual})
    pred["good"] = actual + rng.normal(0, 1, len(idx))
    pred["bad"] = actual + rng.normal(0, 4, len(idx))
    boot = evaluation.DayBootstrap(pred, ["good", "bad"], n=300)
    lo, hi = boot.ci("good")["MAE"]
    assert lo < boot.point("good")["MAE"] < hi
    s, (slo, shi) = boot.skill_ci("good", "bad")
    assert slo < s < shi and slo > 0.4
    assert boot.share_better("good", "bad") > 0.95


def test_mstl_baseline_has_one_row_per_local_hour_across_clock_changes():
    from energy import baselines

    idx = pd.date_range("2025-01-01", periods=24 * 120, freq="h", tz="UTC")
    y = pd.Series(100 + np.sin(np.arange(len(idx)) * 2 * np.pi / 24), index=idx)
    days = pd.date_range("2025-03-29", "2025-04-02", freq="D")
    p = baselines.mstl_day_ahead(y, days, BE, window_days=28)
    assert not p.index.has_duplicates and len(p) == 5 * 24 - 1  # 23-hour day on 30 March
    assert (p - y.reindex(p.index)).abs().max() < 1.0  # a pure daily cycle is recovered
