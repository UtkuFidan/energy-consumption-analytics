"""Customer load profiling: normalised weekday/weekend daily profiles -> archetypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


def client_profiles(hourly: pd.DataFrame, min_active: float = 0.95) -> pd.DataFrame:
    """48-dim profile per client: mean weekday and weekend hourly load / the client's mean load.

    Clients that report zero for more than ``1 - min_active`` of the window are dropped.
    """
    active = (hourly > 0).mean() >= min_active
    h = hourly.loc[:, active]
    scale = h.mean()
    weekend = h.index.day_of_week >= 5
    wd = h[~weekend].groupby(h.index[~weekend].hour).mean() / scale
    we = h[weekend].groupby(h.index[weekend].hour).mean() / scale
    wd.index = [f"wd_{i:02d}" for i in wd.index]
    we.index = [f"we_{i:02d}" for i in we.index]
    return pd.concat([wd, we]).T.dropna()


@dataclass
class Archetypes:
    labels: pd.Series
    centroids: pd.DataFrame
    selection: pd.DataFrame  # k, silhouette, inertia, Davies-Bouldin, stability
    k: int
    ari_vs_ward: float
    summary: pd.DataFrame


def stability(X: np.ndarray, k: int, seed: int, reps: int = 20, frac: float = 0.8) -> float:
    """Mean adjusted Rand index between a full fit and fits on random 80% subsamples."""
    rng = np.random.default_rng(seed)
    full = KMeans(k, n_init=10, random_state=seed).fit(X)
    scores = []
    for _ in range(reps):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        sub = KMeans(k, n_init=10, random_state=seed).fit(X[idx])
        scores.append(adjusted_rand_score(full.predict(X[idx]), sub.labels_))
    return float(np.mean(scores))


def fit_archetypes(
    profiles: pd.DataFrame,
    k_range=range(3, 9),
    mean_kw: pd.Series | None = None,
    seed: int = 7,
    min_share: float = 0.02,
) -> Archetypes:
    X = profiles.to_numpy()
    sel = []
    for k in k_range:
        fit = KMeans(k, n_init=10, random_state=seed).fit(X)
        lab = fit.labels_
        smallest = np.bincount(lab).min() / len(lab)
        sel.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(X, lab)),
                "inertia": float(fit.inertia_),
                "davies_bouldin": float(davies_bouldin_score(X, lab)),
                "stability_ari": stability(X, k, seed),
                "smallest_share": float(smallest),
                "eligible": bool(smallest >= min_share),
            }
        )
    sel = pd.DataFrame(sel)
    ok = sel[sel["eligible"]]  # skip k that only isolates a handful of outlier clients
    k = int((ok if len(ok) else sel).sort_values("silhouette").iloc[-1]["k"])
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    # order archetypes by size so labels are stable and readable
    order = pd.Series(km.labels_).value_counts().index
    remap = {old: new for new, old in enumerate(order)}
    labels = pd.Series([remap[v] for v in km.labels_], index=profiles.index, name="archetype")
    cent = pd.DataFrame(km.cluster_centers_, columns=profiles.columns)
    cent = cent.iloc[list(order)].reset_index(drop=True)
    rows = []
    for a in range(k):
        wd = cent.loc[a, [c for c in cent if c.startswith("wd_")]].to_numpy()
        we = cent.loc[a, [c for c in cent if c.startswith("we_")]].to_numpy()
        members = labels.index[labels == a]
        rows.append(
            {
                "archetype": a,
                "clients": len(members),
                "share_%": 100 * len(members) / len(labels),
                "weekday_peak_hour": int(wd.argmax()),
                "peak_to_mean": float(wd.max()),
                "night_to_day": float(wd[0:6].mean() / wd[9:18].mean()),
                "weekend_to_weekday": float(we.mean() / wd.mean()),
                "mean_kw": float(mean_kw.loc[members].mean()) if mean_kw is not None else np.nan,
            }
        )
    return Archetypes(
        labels, cent, sel, k, float(adjusted_rand_score(km.labels_, ward)),
        pd.DataFrame(rows).set_index("archetype"),
    )  # fmt: skip
