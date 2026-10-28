from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy import forecast, markets

FIX = Path(__file__).parent / "fixtures"


def test_elia_csv_parsing_on_the_export_layout():
    q = markets.parse_elia_csv(FIX / "elia_sample.csv")
    assert list(q.columns) == ["load", "da_fc", "da_p10", "da_p90", "wa_fc", "latest_fc"]
    assert str(q.index.tz) == "UTC" and q.index.is_monotonic_increasing
    assert not q.index.has_duplicates  # the duplicated quarter and the PT60M row are gone
    assert len(q) == 21
    assert q.index[0] == pd.Timestamp("2025-03-29 22:00", tz="UTC")
    assert q.loc["2025-03-29 22:15", "load"] == 7100.0
    assert np.isnan(q.loc["2025-03-30 00:30", "load"])  # empty measurement stays NaN
    assert (q["da_p10"] < q["da_fc"]).all() and (q["da_fc"] < q["da_p90"]).all()


def test_elia_hourly_needs_all_four_quarters():
    q = markets.parse_elia_csv(FIX / "elia_sample.csv")
    h = markets.elia_hourly(q)
    assert h.loc["2025-03-29 22:00", "load"] == pytest.approx((7000 + 7100 + 7050 + 7200) / 4)
    assert np.isnan(h.loc["2025-03-30 00:00", "load"])  # one quarter missing -> hour missing
    assert h.loc["2025-03-30 00:00", "da_fc"] > 0  # forecasts are complete for that hour


def test_energinet_parsing_keeps_last_duplicate():
    s = markets.parse_energinet(FIX / "energinet_sample.json")
    assert list(s.values) == [2500.5, 2401.0]
    assert str(s.index.tz) == "UTC"


def test_open_meteo_multi_point_payload():
    payload = [
        {"hourly": {"time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                    "temperature_2m": [1.0, 2.0], "shortwave_radiation": [0, 0],
                    "cloud_cover": [90, 80], "wind_speed_10m": [10, 12]}},
        {"hourly": {"time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                    "temperature_2m": [3.0, 4.0], "shortwave_radiation": [0, 0],
                    "cloud_cover": [70, 60], "wind_speed_10m": [8, 9]}},
    ]  # fmt: skip
    df = markets.parse_open_meteo(payload, ["bru", "ant"])
    assert str(df.index.tz) == "UTC"
    assert df.loc["2025-01-01 01:00", "temperature_2m_ant"] == 4.0
    w = markets.weather_inputs(df, ["bru", "ant"])
    assert w["temp"].tolist() == [2.0, 3.0]
    assert w["temp_first"].tolist() == [1.0, 2.0]


def test_gate_weather_picks_a_forecast_that_existed_at_gate_closure():
    idx = pd.date_range("2025-06-01", periods=48, freq="h", tz="UTC")
    stitched = pd.DataFrame({"temperature_2m_bru": 0.0}, index=idx)
    prev = pd.DataFrame(
        {"temperature_2m_previous_day1_bru": 1.0, "temperature_2m_previous_day2_bru": 2.0},
        index=idx,
    )
    lead = forecast.horizon_hours(idx, forecast.BELGIUM)
    out = markets.gate_weather(stitched, prev, lead)["temperature_2m_bru"]
    # horizons up to 19 h may use the day-1 run; later hours must fall back to day-2
    assert set(out[lead <= 19]) == {1.0} and set(out[lead > 19]) == {2.0}
    # no previous-run data: keep the stitched series
    assert markets.gate_weather(stitched, None, lead).equals(stitched)
