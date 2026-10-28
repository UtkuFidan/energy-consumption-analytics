import numpy as np
import pandas as pd
import pytest

from energy import synth, theft


@pytest.fixture(scope="module")
def injected():
    X = synth.daily_meters(n=200, days=400)
    Xi, kinds = theft.inject_theft(X, frac=0.15, seed=3)
    return X, Xi, kinds, theft.compute_features(Xi)


def test_injection_only_touches_chosen_meters(injected):
    X, Xi, kinds, _ = injected
    clean = kinds.index[kinds == ""]
    pd.testing.assert_frame_equal(X[clean], Xi[clean])
    assert set(kinds[kinds != ""]) == set(theft.PATTERNS)
    assert (kinds != "").sum() == 30


def test_drop_features_fire_on_sudden_drops(injected):
    _, _, kinds, F = injected
    clean = F[kinds == ""]
    drops = F[kinds == "sudden_drop"]
    assert (drops["step_drop"] > 0.4).all()
    assert clean["step_drop"].quantile(0.95) < 0.3


def test_flat_and_zero_features_fire_on_flatline(injected):
    _, _, kinds, F = injected
    assert (F.loc[kinds == "flatline", "longest_flat_run"] > 100).all()
    assert F.loc[kinds == "", "longest_flat_run"].max() < 5


def test_switching_features_fire_on_periodic_bypass(injected):
    _, _, kinds, F = injected
    assert F.loc[kinds == "periodic_bypass", "switch_rate"].min() > F.loc[
        kinds == "", "switch_rate"
    ].quantile(0.95)


def test_zero_period_and_reason_codes(injected):
    _, Xi, kinds, _ = injected
    Xz = Xi.copy()
    Xz.iloc[250:290, 0] = 0.0
    F = theft.compute_features(Xz)
    assert F.iloc[0]["longest_zero_run"] == 40
    codes = theft.reason_codes(F, theft.fit_thresholds(F))
    assert "ZERO_PERIODS" in codes.iloc[0]
    flat = kinds.index[kinds == "flatline"]
    assert codes.loc[flat].str.contains("FLAT_LINE").all()
    drops = kinds.index[kinds == "sudden_drop"]
    assert codes.loc[drops].str.contains("SUSTAINED_DROP").any()


def test_weekly_pattern_break():
    X = synth.daily_meters(n=40, days=400, seed=5)
    X = X.copy()
    # meter 0 loses its weekly rhythm in the second half
    X.iloc[200:, 0] = X.iloc[200:, 0].mean()
    F = theft.compute_features(X)
    assert F["weekly_break"].idxmax() == X.columns[0]


def test_precision_at_k_and_budget():
    y = pd.Series([1, 0, 1, 0, 0, 1], index=list("abcdef"))
    s = pd.Series([0.9, 0.8, 0.7, 0.2, 0.1, 0.05], index=y.index)
    assert theft.precision_at_k(y, s, 2) == 0.5
    assert theft.precision_at_k(y, s, 3) == pytest.approx(2 / 3)
    assert theft.inspection_budget(10, 8, 5) == 400
    ev = theft.evaluate(y, {"m": s}, {"3": 3})
    assert ev.loc["m", "PR_AUC"] > 0.7 and ev.loc["m", "lift@3"] == pytest.approx((2 / 3) / 0.5)


def test_models_beat_base_rate_on_injected_patterns(injected):
    _, _, kinds, F = injected
    y = (kinds != "").astype(int)
    iso = theft.isolation_forest_scores(F)
    xg, imp = theft.xgb_oof_scores(F, y, n_splits=3)
    ev = theft.evaluate(y, {"iso": iso, "xgb": xg}, {"15": 15})
    assert ev.loc["xgb", "PR_AUC"] > 3 * y.mean()
    assert ev.loc["iso", "ROC_AUC"] > 0.7
    assert len(imp) == len(theft.FEATURES)


def test_inspection_list_has_reason_codes(injected):
    _, _, kinds, F = injected
    score = F["step_drop"] + F["longest_flat_run"] / 100
    lst = theft.build_inspection_list(F, score, theft.fit_thresholds(F), 10)
    assert list(lst["rank"]) == list(range(1, 11))
    assert lst["reason_codes"].str.len().min() > 0
    assert (kinds.loc[lst.index] != "").mean() >= 0.8


def test_new_shape_features_flag_a_fading_meter():
    X = synth.daily_meters(n=60, days=420, seed=9).copy()
    X.iloc[300:, 3] *= 0.2  # meter 3 collapses for the last 120 days
    F = theft.compute_features(X)
    assert F.loc[X.columns[3], "recent_ratio"] < 0.5
    assert F.loc[X.columns[3], "late_vs_early"] < F["late_vs_early"].quantile(0.05)
    assert F["recent_ratio"].idxmin() == X.columns[3]
    assert set(theft.FEATURES) <= set(F.columns)
    assert np.isfinite(F[theft.FEATURES].to_numpy()).all()


def test_tuned_model_and_bootstrap_on_injected_patterns(injected):
    _, _, kinds, F = injected
    y = (kinds != "").astype(int)
    oof, fold_ap, chosen, imp = theft.xgb_nested_oof(F, y, n_outer=3, n_trials=2)
    assert len(fold_ap) == len(chosen) == 3 and "n_estimators" in chosen[0]
    ev = theft.bootstrap_metrics(y, {"tuned": oof}, {"15": 15}, n_boot=50)
    r = ev.loc["tuned"]
    assert r["PR_AUC_lo"] <= r["PR_AUC"] <= r["PR_AUC_hi"]
    assert r["P@15_lo"] <= r["P@15"] <= r["P@15_hi"]
    assert r["PR_AUC"] > 3 * y.mean()
