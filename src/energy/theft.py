"""Electricity theft / non-technical-loss detection on daily meter data.

Input everywhere: ``X`` with one row per day and one column per meter (NaN = missing).
Features are computed from the consumption series only, never from the labels.
"""

from __future__ import annotations

import warnings

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from . import dq

optuna.logging.set_verbosity(optuna.logging.WARNING)

BLOCK = 30  # days per comparison block
SEED = 7

FEATURES = [
    "level_log", "cv", "nan_frac", "zero_frac", "longest_zero_run", "longest_flat_run",
    "step_drop", "recent_drop", "block_drop", "peer_level_z", "peer_rel_min", "peer_rel_last",
    "weekly_break", "switch_rate", "low_frac",
    "recent_ratio", "late_vs_early", "trend_slope", "p10_rel", "p90_rel", "spike_frac", "ac7",
    "weekend_ratio", "zero_runs", "longest_nan_run", "peer_cv_z", "peer_drop_z",
]  # fmt: skip


def _block_means(a: np.ndarray, block: int) -> np.ndarray:
    n_blocks = a.shape[0] // block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[: n_blocks * block].reshape(n_blocks, block, -1), axis=1)


def _step_ratio(bm: np.ndarray, min_side: int = 3) -> np.ndarray:
    """Min over split points of mean(blocks after) / mean(blocks before): a level shift detector."""
    nb = bm.shape[0]
    val = ~np.isnan(bm)
    s = np.cumsum(np.where(val, bm, 0.0), axis=0)
    c = np.cumsum(val, axis=0)
    best = np.full(bm.shape[1], np.inf)
    for b in range(min_side, nb - min_side + 1):
        pre_n, post_n = c[b - 1], c[-1] - c[b - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            pre = s[b - 1] / pre_n
            post = (s[-1] - s[b - 1]) / post_n
            ratio = np.where((pre > 1e-6) & (pre_n > 0) & (post_n > 0), post / pre, np.inf)
        best = np.minimum(best, ratio)
    return np.where(np.isinf(best), 1.0, best)


def _block_dip(bm: np.ndarray, lookback: int = 6) -> np.ndarray:
    """Worst single block relative to the mean of the previous ``lookback`` blocks."""
    worst = np.ones(bm.shape[1])
    for b in range(3, bm.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prev = np.nanmean(bm[max(0, b - lookback) : b], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(prev > 1e-6, bm[b] / prev, 1.0)
        worst = np.minimum(worst, np.nan_to_num(r, nan=1.0))
    return worst


def peer_groups(level_log: np.ndarray, shape: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """Peer clusters from consumption level and seasonal shape (no labels, no theft info)."""
    z = (level_log - np.nanmean(level_log)) / (np.nanstd(level_log) + 1e-9)
    Z = np.column_stack([z, np.nan_to_num(shape, nan=1.0)])
    k = max(1, min(k, len(Z) // 5))
    return MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit_predict(Z)


def _weekly_break(a: np.ndarray, chunk: int = 91) -> np.ndarray:
    """Max L1 distance of a chunk's day-of-week profile from the meter's median profile."""
    n_chunks = a.shape[0] // chunk
    if n_chunks < 2:
        return np.zeros(a.shape[1])
    prof = []
    for c in range(n_chunks):
        seg = a[c * chunk : (c + 1) * chunk]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dow = np.stack([np.nanmean(seg[j::7], axis=0) for j in range(7)])
            prof.append(dow / (np.nanmean(dow, axis=0) + 1e-6))
    prof = np.stack(prof)  # chunks x 7 x meters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(prof, axis=0)
        dist = np.nanmax(np.nansum(np.abs(prof - med), axis=1), axis=0)
    return np.nan_to_num(dist, nan=0.0)


def _longest_nan_run(missing: np.ndarray) -> np.ndarray:
    run = np.zeros(missing.shape[1], dtype=np.int64)
    best = np.zeros_like(run)
    for row in missing:
        run = (run + 1) * row
        np.maximum(best, run, out=best)
    return best


def _peer_z(f: pd.DataFrame, col: str, groups: np.ndarray) -> np.ndarray:
    """Robust z-score of a feature inside the meter's peer group."""
    g = f[col].groupby(groups)
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    return ((f[col] - med) / (1.4826 * mad + 1e-3)).clip(-50, 50).to_numpy()


def _shape_features(
    a: np.ndarray, days: pd.DatetimeIndex, med_pos: np.ndarray, index: pd.Index
) -> pd.DataFrame:
    """Drop ratios and distribution shape of the daily series, all relative to the meter itself."""
    T = a.shape[0]
    third = T // 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_all = np.nanmean(a, axis=0)
        early, late = np.nanmean(a[:third], axis=0), np.nanmean(a[-third:], axis=0)
        recent, prior = np.nanmean(a[-90:], axis=0), np.nanmean(a[-270:-90], axis=0)
        q10, q90 = np.nanpercentile(a, [10, 90], axis=0)
        wk = np.asarray(days.day_of_week >= 5)
        wkend, wkday = np.nanmean(a[wk], axis=0), np.nanmean(a[~wk], axis=0)
        # slope of the daily series (per year, relative to its mean), NaN-aware least squares
        t = np.arange(T, dtype=np.float64)[:, None]
        ok = ~np.isnan(a)
        n = np.maximum(ok.sum(axis=0), 2)
        tm = (t * ok).sum(axis=0) / n
        am = np.nansum(a, axis=0) / n
        cov = (((t - tm) * (np.nan_to_num(a) - am)) * ok).sum(axis=0)
        var = (((t - tm) ** 2) * ok).sum(axis=0)
        slope = cov / np.maximum(var, 1e-9) * 365 / np.maximum(mean_all, 1e-6)
        filled = np.where(ok, a, am[None, :])
        c = filled - filled.mean(axis=0)
        ac7 = (c[7:] * c[:-7]).sum(axis=0) / np.maximum((c**2).sum(axis=0), 1e-9)
        spikes = np.nanmean(
            np.where(ok, a > 3 * np.maximum(med_pos, 1e-6)[None, :], np.nan), axis=0
        )

    def ratio(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip(np.nan_to_num(np.where(y > 1e-6, x / y, 1.0), nan=1.0), 0, 20)

    return pd.DataFrame(
        {
            "recent_ratio": ratio(recent, prior),
            "late_vs_early": ratio(late, early),
            "trend_slope": np.nan_to_num(slope),
            "p10_rel": ratio(q10, med_pos),
            "p90_rel": ratio(q90, med_pos),
            "spike_frac": np.nan_to_num(spikes),
            "ac7": ac7,
            "weekend_ratio": ratio(wkend, wkday),
        },
        index=index,
    )


def compute_features(X: pd.DataFrame, n_peer_groups: int = 8) -> pd.DataFrame:
    a = X.to_numpy(dtype=np.float64)
    valid = ~np.isnan(a)
    n_valid = np.maximum(valid.sum(axis=0), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nan_to_num(np.nanmean(a, axis=0))
        sd = np.nan_to_num(np.nanstd(a, axis=0))
        med_pos = np.nan_to_num(np.nanmedian(np.where(a > 0, a, np.nan), axis=0))
    f = pd.DataFrame(index=X.columns)
    f["level_log"] = np.log1p(np.maximum(level, 0))
    f["cv"] = sd / np.maximum(level, 1e-6)
    f["nan_frac"] = 1 - valid.sum(axis=0) / a.shape[0]
    f["zero_frac"] = ((a <= 0) & valid).sum(axis=0) / n_valid

    zr = dq.find_zero_runs(X, min_run=14)
    st = dq.find_stuck(X, min_run=7)
    f["longest_zero_run"] = zr["longest_zero_run"].to_numpy()
    f["longest_flat_run"] = st["longest_stuck"].to_numpy()

    bm = _block_means(a, BLOCK)
    f["step_drop"] = np.clip(1 - _step_ratio(bm), 0, 1)
    f["block_drop"] = np.clip(1 - _block_dip(bm), 0, 1)
    f["recent_drop"] = (
        np.clip(1 - _step_ratio(bm[-6:], min_side=3), 0, 1) if bm.shape[0] >= 6 else 0.0
    )

    own = np.where(level > 1e-6, level, np.nan)
    shape = bm / own  # seasonal shape, 1 = own average
    pg = peer_groups(f["level_log"].to_numpy(), shape.T, n_peer_groups)
    f["peer_group"] = pg
    grp = f.groupby("peer_group")["level_log"]
    med, mad = grp.transform("median"), grp.transform(lambda s: (s - s.median()).abs().median())
    f["peer_level_z"] = (f["level_log"] - med) / (1.4826 * mad + 1e-3)

    shape_df = pd.DataFrame(shape.T)
    peer_med = shape_df.groupby(pg).transform("median").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(peer_med > 1e-6, shape_df.to_numpy() / peer_med, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["peer_rel_min"] = np.nan_to_num(np.nanmin(rel, axis=1), nan=1.0)
        f["peer_rel_last"] = np.nan_to_num(np.nanmean(rel[:, -3:], axis=1), nan=1.0)

    f["weekly_break"] = _weekly_break(a)
    f = f.join(_shape_features(a, X.index, med_pos, f.index))
    f["zero_runs"] = dq.find_zero_runs(X, min_run=7)["zero_runs"].to_numpy()
    f["longest_nan_run"] = _longest_nan_run(~valid)
    f["peer_cv_z"] = _peer_z(f, "cv", pg)
    f["peer_drop_z"] = _peer_z(f, "step_drop", pg)
    low = (a < 0.25 * med_pos[None, :]) & valid
    f["low_frac"] = low.sum(axis=0) / n_valid
    f["switch_rate"] = (low[1:] != low[:-1]).sum(axis=0) / a.shape[0] * 100
    return f.drop(columns="peer_group")


# ---------------------------------------------------------------------------
# Reason codes and inspection list
# ---------------------------------------------------------------------------

# code -> (feature, direction, population quantile, absolute floor). A meter gets the code when
# its feature is beyond both the quantile cut-off and the floor.
RULES = {
    "SUSTAINED_DROP": ("step_drop", "high", 0.95, 0.30),
    "RECENT_DROP": ("recent_drop", "high", 0.95, 0.30),
    "BELOW_PEERS": ("peer_rel_last", "low", 0.05, 0.50),
    "ZERO_PERIODS": ("longest_zero_run", "high", 0.95, 14),
    "FLAT_LINE": ("longest_flat_run", "high", 0.95, 7),
    "WEEKLY_PATTERN_BREAK": ("weekly_break", "high", 0.95, 1.0),
    "ON_OFF_SWITCHING": ("switch_rate", "high", 0.95, 4.0),
    "LOW_LEVEL_VS_PEERS": ("peer_level_z", "low", 0.05, -1.5),
    "HIGH_LEVEL_VS_PEERS": ("peer_level_z", "high", 0.95, 1.5),
    "ERRATIC_USAGE": ("cv", "high", 0.95, 1.0),
    "DATA_GAPS": ("nan_frac", "high", 0.95, 0.10),
}


def fit_thresholds(F: pd.DataFrame) -> dict[str, float]:
    out = {}
    for code, (col, direction, q, floor) in RULES.items():
        cut = F[col].quantile(q)
        out[code] = max(cut, floor) if direction == "high" else min(cut, floor)
    return out


def reason_codes(F: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    hits = pd.DataFrame(index=F.index)
    for code, (col, direction, _, _) in RULES.items():
        hits[code] = (
            F[col] >= thresholds[code] if direction == "high" else F[col] <= thresholds[code]
        )
    names = hits.columns.to_numpy()
    joined = [";".join(names[row]) or "MODEL_SCORE_ONLY" for row in hits.to_numpy(dtype=bool)]
    return pd.Series(joined, index=F.index, name="reason_codes")


def build_inspection_list(
    F: pd.DataFrame, score: pd.Series, thresholds: dict, k: int
) -> pd.DataFrame:
    top = score.sort_values(ascending=False).head(k)
    codes = reason_codes(F.loc[top.index], thresholds)
    cols = ["step_drop", "peer_rel_min", "longest_zero_run", "longest_flat_run", "weekly_break"]
    out = pd.concat([top.rename("risk_score"), codes, F.loc[top.index, cols].round(3)], axis=1)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.index.name = "meter"
    return out


# ---------------------------------------------------------------------------
# Models and evaluation
# ---------------------------------------------------------------------------


BASE_FEATURES = FEATURES[:15]  # first version of the feature set, kept for the ablation row


def _impute(F: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or FEATURES
    return F[cols].replace([np.inf, -np.inf], np.nan).fillna(F[cols].median())


def isolation_forest_scores(F: pd.DataFrame, seed: int = SEED,
                            cols: list[str] | None = None) -> pd.Series:  # fmt: skip
    """Unsupervised anomaly score (higher = more anomalous)."""
    Z = RobustScaler().fit_transform(_impute(F, cols))
    iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(Z)
    return pd.Series(-iso.score_samples(Z), index=F.index, name="iforest")


def _xgb_clf(pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, scale_pos_weight=pos_weight, eval_metric="aucpr", random_state=SEED,
        n_jobs=-1, tree_method="hist",
    )  # fmt: skip


def xgb_oof_scores(F: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = SEED,
                   cols: list[str] | None = None):  # fmt: skip
    """Out-of-fold probabilities from stratified CV, plus gain importances from a full fit."""
    Z, yv = _impute(F, cols), y.to_numpy()
    oof = np.zeros(len(Z))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Z, yv):
        pw = (yv[tr] == 0).sum() / max((yv[tr] == 1).sum(), 1)
        oof[te] = _xgb_clf(pw).fit(Z.iloc[tr], yv[tr]).predict_proba(Z.iloc[te])[:, 1]
    pw = (yv == 0).sum() / max((yv == 1).sum(), 1)
    imp = pd.Series(_xgb_clf(pw).fit(Z, yv).feature_importances_, index=Z.columns)
    return pd.Series(oof, index=F.index, name="xgb"), imp.sort_values(ascending=False)


def _xgb_from(params: dict, pos_weight: float) -> xgb.XGBClassifier:
    p = dict(params)
    mult = p.pop("pos_weight_exp", 1.0)  # scale_pos_weight = (neg/pos) ** mult
    return xgb.XGBClassifier(
        scale_pos_weight=pos_weight**mult, eval_metric="aucpr", random_state=SEED,
        n_jobs=-1, tree_method="hist", **p,
    )  # fmt: skip


def tune_xgb(Z: pd.DataFrame, y: np.ndarray, n_trials: int, inner_splits: int = 3,
             seed: int = SEED) -> dict:  # fmt: skip
    """Optuna (TPE) on stratified inner CV, maximising mean average precision."""
    inner = list(StratifiedKFold(inner_splits, shuffle=True, random_state=seed).split(Z, y))

    def objective(t: optuna.Trial) -> float:
        params = {
            "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 8),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 30, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": t.suggest_float("gamma", 1e-3, 5.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
            "pos_weight_exp": t.suggest_float("pos_weight_exp", 0.0, 1.0),
        }
        scores = []
        for i, (tr, va) in enumerate(inner):
            pw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            m = _xgb_from(params, pw).fit(Z.iloc[tr], y[tr])
            scores.append(average_precision_score(y[va], m.predict_proba(Z.iloc[va])[:, 1]))
            t.report(float(np.mean(scores)), i)
            if t.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
    )  # fmt: skip
    study.optimize(objective, n_trials=n_trials)
    return {**study.best_params, "inner_ap": study.best_value}


def xgb_nested_oof(F: pd.DataFrame, y: pd.Series, n_outer: int = 5, n_trials: int = 20,
                   seed: int = SEED):  # fmt: skip
    """Out-of-fold probabilities where each outer fold's model was tuned only on its training part.

    Hyperparameters are searched with Optuna on a 3-fold stratified split inside the outer
    training set, so the outer fold never influences its own model or its own settings.
    Returns scores, per-fold PR-AUC, the chosen parameters per fold and gain importances.
    """
    Z, yv = _impute(F), y.to_numpy()
    oof = np.zeros(len(Z))
    chosen, fold_ap = [], []
    imp = pd.Series(0.0, index=Z.columns)
    for tr, te in StratifiedKFold(n_outer, shuffle=True, random_state=seed).split(Z, yv):
        best = tune_xgb(Z.iloc[tr], yv[tr], n_trials, seed=seed)
        params = {k: v for k, v in best.items() if k != "inner_ap"}
        pw = (yv[tr] == 0).sum() / max((yv[tr] == 1).sum(), 1)
        model = _xgb_from(params, pw).fit(Z.iloc[tr], yv[tr])
        oof[te] = model.predict_proba(Z.iloc[te])[:, 1]
        fold_ap.append(float(average_precision_score(yv[te], oof[te])))
        chosen.append(best)
        imp += pd.Series(model.feature_importances_, index=Z.columns) / n_outer
    return (
        pd.Series(oof, index=F.index, name="xgb_tuned"),
        fold_ap,
        chosen,
        imp.sort_values(ascending=False),
    )


def bootstrap_metrics(y: pd.Series, scores: dict[str, pd.Series], budgets: dict[str, int],
                      n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:  # fmt: skip
    """Point estimates and 95% percentile-bootstrap CIs (resampling meters) for PR-AUC and P@k."""
    rng = np.random.default_rng(seed)
    yv = y.to_numpy()
    n = len(yv)
    draws = [rng.integers(0, n, n) for _ in range(n_boot)]
    rows = []
    for name, s in scores.items():
        sv = s.loc[y.index].to_numpy()

        def stats(idx: np.ndarray, sv=sv) -> dict:
            yy, ss = yv[idx], sv[idx]
            top = np.argsort(-ss, kind="stable")
            out = {"PR_AUC": average_precision_score(yy, ss)}
            for lab, k in budgets.items():
                out[f"P@{lab}"] = yy[top[:k]].mean()
            return out

        point = stats(np.arange(n))
        boots = pd.DataFrame([stats(d) for d in draws])
        r = {"model": name, "ROC_AUC": float(roc_auc_score(yv, sv))}
        for key, v in point.items():
            r[key] = float(v)
            r[f"{key}_lo"] = float(boots[key].quantile(0.025))
            r[f"{key}_hi"] = float(boots[key].quantile(0.975))
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def inspection_budget(inspectors: int, meters_per_inspector_day: int, days: int) -> int:
    return inspectors * meters_per_inspector_day * days


def precision_at_k(y: pd.Series, score: pd.Series, k: int) -> float:
    top = score.sort_values(ascending=False).head(k).index
    return float(y.loc[top].mean())


def evaluate(y: pd.Series, scores: dict[str, pd.Series], budgets: dict[str, int]) -> pd.DataFrame:
    rows = []
    base = float(y.mean())
    for name, s in scores.items():
        r = {
            "model": name,
            "PR_AUC": float(average_precision_score(y, s)),
            "ROC_AUC": float(roc_auc_score(y, s)),
        }
        for lab, k in budgets.items():
            p = precision_at_k(y, s, k)
            r[f"P@{lab}"] = p
            r[f"lift@{lab}"] = p / base
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Synthetic theft injection (documented patterns)
# ---------------------------------------------------------------------------

PATTERNS = ("sudden_drop", "flatline", "periodic_bypass")


def inject_theft(
    X: pd.DataFrame, frac: float = 0.15, seed: int = SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """Inject theft-like patterns into a random ``frac`` of meters.

    - sudden_drop: from a random day (30-70 % into the series) consumption is scaled by 0.1-0.5
    - flatline: from a random day the meter repeats one value (a stuck register) at 20-60 % of
      its previous 4-week average
    - periodic_bypass: from a random day, every 7-14 days a 3-5 day window drops to 5 %
    Returns the modified frame and a Series of pattern names ('' for clean meters).
    """
    rng = np.random.default_rng(seed)
    out = X.copy()
    T = len(X)
    kinds = pd.Series("", index=X.columns, name="injected")
    chosen = rng.choice(X.columns, size=int(round(frac * X.shape[1])), replace=False)
    for i, m in enumerate(chosen):
        kind = PATTERNS[i % len(PATTERNS)]
        s = int(rng.uniform(0.3, 0.7) * T)
        v = out[m].to_numpy(dtype=np.float64).copy()
        if kind == "sudden_drop":
            v[s:] *= rng.uniform(0.1, 0.5)
        elif kind == "flatline":
            v[s:] = np.nanmean(v[max(0, s - 28) : s]) * rng.uniform(0.2, 0.6)
        else:
            cycle, width = int(rng.integers(7, 15)), int(rng.integers(3, 6))
            for t in range(s, T, cycle):
                v[t : t + width] *= 0.05
        out[m] = v
        kinds[m] = kind
    return out, kinds
