import numpy as np
import pandas as pd

from energy import profiling


def test_two_shapes_are_separated():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2014-01-01", periods=24 * 28, freq="h")
    h = idx.hour.to_numpy()
    shapes = {
        "day": 1 + 0.8 * np.exp(-((h - 13) ** 2) / 8),
        "eve": 1 + 0.8 * np.exp(-((h - 20) ** 2) / 8),
    }
    cols = {}
    for i in range(30):
        s = shapes["day" if i < 15 else "eve"]
        cols[f"C{i}"] = 100 * s * (1 + 0.02 * rng.standard_normal(len(idx)))
    hourly = pd.DataFrame(cols, index=idx)
    prof = profiling.client_profiles(hourly)
    assert prof.shape == (30, 48)
    arch = profiling.fit_archetypes(prof, k_range=range(2, 5), mean_kw=hourly.mean(), min_share=0.1)
    assert arch.k == 2
    a, b = arch.labels.iloc[:15], arch.labels.iloc[15:]
    assert a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] != b.iloc[0]
    assert set(arch.summary["weekday_peak_hour"]) == {13, 20}


def test_inactive_clients_are_dropped():
    idx = pd.date_range("2014-01-01", periods=24 * 14, freq="h")
    df = pd.DataFrame({"on": 1.0, "off": 0.0}, index=idx)
    assert list(profiling.client_profiles(df).index) == ["on"]
