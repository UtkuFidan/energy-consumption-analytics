import numpy as np
import pandas as pd

from energy import dq


def _frame(n_meters=4, days=10):
    idx = pd.date_range("2014-03-25", periods=days * 24, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.uniform(1, 5, (len(idx), n_meters)), index=idx, columns=list("ABCD"))


def test_gaps_and_duplicates():
    df = _frame()
    df = df.drop(df.index[[10, 11, 50]])
    df = pd.concat([df, df.iloc[[5]].assign(A=99.0)]).sort_index()
    g = dq.find_gaps(df, "h")
    assert g["missing_timestamps"] == 3
    d = dq.find_duplicates(df)
    assert d["duplicate_timestamps"] == 1 and d["conflicting_timestamps"] == 1


def test_negative_values():
    df = _frame()
    df.iloc[3, 1] = -2.0
    df.iloc[7, 1] = -1.0
    neg = dq.find_negatives(df)
    assert neg["B"] == 2 and neg["A"] == 0


def test_stuck_meter_and_zero_run_and_leading_zeros():
    df = _frame()
    df.iloc[20:40, 0] = 3.3  # 20 identical readings
    df.iloc[:30, 1] = 0.0  # not yet commissioned
    df.iloc[100:130, 2] = 0.0  # outage after commissioning
    stuck = dq.find_stuck(df, min_run=12)
    assert stuck.loc["A", "longest_stuck"] == 20 and stuck.loc["A", "stuck_runs"] == 1
    assert stuck.loc["D", "stuck_runs"] == 0
    assert dq.leading_zero_steps(df)["B"] == 30
    zr = dq.find_zero_runs(df, min_run=12)
    assert zr.loc["C", "longest_zero_run"] == 30 and zr.loc["C", "zero_runs"] == 1
    assert zr.loc["B", "zero_runs"] == 0  # leading zeros are not an outage


def test_dst_anomalies_detected_on_both_clock_changes():
    idx = pd.date_range("2014-03-27", "2014-11-02", freq="h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.uniform(1, 2, (len(idx), 5)), index=idx, columns=list("ABCDE"))
    spring, autumn = pd.Timestamp("2014-03-30 01:00"), pd.Timestamp("2014-10-26 01:00")
    df.loc[spring] = 0.0  # missing hour reported as zero
    df.loc[autumn] *= 2  # repeated hour folded into one
    out = dq.dst_anomalies(df).set_index("kind")
    assert out.loc["spring", "zero_share"] == 1.0
    assert out.loc["autumn", "median_ratio"] > 1.6


def test_quality_report_summary_and_markdown():
    df = _frame()
    df.iloc[5, 0] = -1
    rep = dq.quality_report(df, dq.DQConfig(freq="h", stuck_min_run=6, zero_min_run=6))
    assert rep["summary"]["meters_with_negatives"] == 1
    assert "Meter data-quality report" in dq.to_markdown(rep)


def test_tso_report_flags_missing_quarters_and_forecast_departures():
    from pathlib import Path

    from energy import markets

    q = markets.parse_elia_csv(Path(__file__).parent / "fixtures" / "elia_sample.csv")
    rep = dq.tso_load_report(q, raw_rows=23)
    assert rep["duplicate_timestamps"] == 2
    assert rep["missing_timestamps"] == 0
    assert rep["missing_measured"] == ["2025-03-30 00:30"]
    assert rep["non_positive"] == 0 and rep["forecast_band_violations"] == 0
    # a 3 GW excursion against the most recent forecast is reported as one episode
    q2 = q.copy()
    q2.loc["2025-03-30 01:00":"2025-03-30 01:30", "load"] += 3000
    eps = dq.tso_load_report(q2, raw_rows=21, z_cut=6.0)["episodes"]
    assert len(eps) == 1 and eps.iloc[0]["quarters"] == 3
    assert "Largest departures" in dq.tso_to_markdown(dq.tso_load_report(q2, 21), "t")
