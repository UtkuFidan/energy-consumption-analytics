"""Forecast scoring: point metrics, skill, pinball loss, interval reports, day-block bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    e = pred - actual
    return {
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e**2).mean())),
        "MAPE_%": float((e.abs() / actual.abs()).mean() * 100),
        "sMAPE_%": float((2 * e.abs() / (actual.abs() + pred.abs())).mean() * 100),
    }


def skill(mae_model: float, mae_ref: float) -> float:
    """1 - MAE/MAE_ref: share of the reference error removed (negative = worse than reference)."""
    return 1.0 - mae_model / mae_ref


def pinball(actual: pd.Series, pred: pd.Series, q: float) -> float:
    d = actual - pred
    return float(np.maximum(q * d, (q - 1) * d).mean())


def interval_report(
    actual: pd.Series, lo: pd.Series, hi: pd.Series, mid: pd.Series, nominal: float = 0.8
) -> dict:
    """Coverage, width and pinball loss of a central interval (q = (1-nominal)/2 and 1-q)."""
    a = (1 - nominal) / 2
    inside = (actual >= lo) & (actual <= hi)
    return {
        "nominal": nominal,
        "coverage": float(inside.mean()),
        "mean_width": float((hi - lo).mean()),
        "width_pct_of_load": float(((hi - lo) / actual).mean() * 100),
        f"pinball_q{round(a * 100):02d}": pinball(actual, lo, a),
        "pinball_q50": pinball(actual, mid, 0.5),
        f"pinball_q{round((1 - a) * 100):02d}": pinball(actual, hi, 1 - a),
    }


def error_by_hour(pred: pd.DataFrame, models: list[str], tz: str | None = None) -> pd.DataFrame:
    idx = pred.index.tz_convert(tz) if tz else pred.index
    err = {m: (pred[m] - pred["actual"]).abs() for m in models}
    return pd.DataFrame(err).groupby(idx.hour).mean()


class DayBootstrap:
    """Moving-block bootstrap with whole days as blocks.

    Hourly errors inside a day are strongly correlated, so resampling hours would give
    intervals that are far too tight. Statistics are recomputed from per-day sums, which
    keeps thousands of replicates cheap.
    """

    def __init__(self, pred: pd.DataFrame, models: list[str], tz: str | None = None, n: int = 2000,
                 seed: int = 11):  # fmt: skip
        idx = pred.index.tz_convert(tz) if tz else pred.index
        keys, self.days = pd.factorize(idx.normalize())
        self.models = models
        a = pred["actual"]
        nd = len(self.days)
        self.S = {}
        for m in models:
            e = (pred[m] - a).to_numpy()
            ok = np.isfinite(e)
            ape = np.abs(e) / np.abs(a.to_numpy())
            cols = np.column_stack([np.abs(e), e**2, ape, np.ones(len(e))])
            cols[~ok] = 0
            S = np.zeros((nd, 4))
            np.add.at(S, keys, cols)
            self.S[m] = S
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, nd, size=(n, nd))
        self.W = np.stack([np.bincount(r, minlength=nd) for r in draws]).astype(float)

    def _stat(self, m: str, W: np.ndarray) -> dict[str, np.ndarray]:
        s = W @ self.S[m]
        return {"MAE": s[:, 0] / s[:, 3], "RMSE": np.sqrt(s[:, 1] / s[:, 3]),
                "MAPE_%": 100 * s[:, 2] / s[:, 3]}  # fmt: skip

    def point(self, m: str) -> dict[str, float]:
        return {k: float(v[0]) for k, v in self._stat(m, np.ones((1, len(self.days)))).items()}

    def ci(self, m: str, level: float = 0.95) -> dict[str, tuple[float, float]]:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return {k: (float(np.quantile(v, lo)), float(np.quantile(v, hi)))
                for k, v in self._stat(m, self.W).items()}  # fmt: skip

    def skill_ci(self, m: str, ref: str, metric: str = "MAE", level: float = 0.95):
        """Point estimate and CI of 1 - metric(m)/metric(ref), paired over the same days."""
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        one = np.ones((1, len(self.days)))
        point = 1 - self._stat(m, one)[metric][0] / self._stat(ref, one)[metric][0]
        draws = 1 - self._stat(m, self.W)[metric] / self._stat(ref, self.W)[metric]
        return float(point), (float(np.quantile(draws, lo)), float(np.quantile(draws, hi)))

    def share_better(self, m: str, ref: str) -> float:
        """Share of days on which m has the lower MAE than ref."""
        a = self.S[m][:, 0] / self.S[m][:, 3]
        b = self.S[ref][:, 0] / self.S[ref][:, 3]
        return float((a < b).mean())
