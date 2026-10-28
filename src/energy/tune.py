"""Optuna tuning of the day-ahead GBMs on a rolling-origin backtest.

Validation folds are whole months that all lie before the test period; each fold is fitted
on everything that precedes it (with the gate-closure cut-off), exactly like the real backtest.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from . import forecast as fc

optuna.logging.set_verbosity(optuna.logging.WARNING)


def validation_folds(index: pd.DatetimeIndex, months: list[str], info: fc.InfoSet) -> list[fc.Fold]:
    folds = []
    for m in months:
        start = pd.Timestamp(m)
        end = start + pd.offsets.MonthBegin(1)
        folds += list(fc.rolling_origin_splits(index, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", info))
    return folds


def _score(make, defaults, params, feats, cols, folds) -> list[float]:
    maes = []
    for fold in folds:
        tr = feats[(feats.index < fold.train_end) & feats["y"].notna()]
        te = feats[(feats.index >= fold.test_start) & (feats.index < fold.test_end)]
        te = te[te["y"].notna()]
        p = fc.fit_predict(make, params, tr, te, cols, defaults)
        maes.append(float(np.mean(np.abs(p - te["y"].to_numpy()))))
    return maes


def _lgb_space(t: optuna.Trial) -> dict:
    return {
        "target": t.suggest_categorical("target", ["level", "diff_w"]),
        "n_estimators": t.suggest_int("n_estimators", 150, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": t.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 150, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _xgb_space(t: optuna.Trial) -> dict:
    return {
        "target": t.suggest_categorical("target", ["level", "diff_w"]),
        "n_estimators": t.suggest_int("n_estimators", 150, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": t.suggest_int("max_depth", 3, 9),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 40, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def tune(kind: str, feats: pd.DataFrame, cols: list[str], folds: list[fc.Fold],
         n_trials: int, seed: int = fc.SEED) -> dict:  # fmt: skip
    """Minimise the mean fold MAE. Returns the best parameters and the baseline comparison."""
    make, defaults, space = {
        "lgbm": (fc.make_lgb, fc.LGB_DEFAULTS, _lgb_space),
        "xgb": (fc.make_xgb, fc.XGB_DEFAULTS, _xgb_space),
    }[kind]

    def objective(trial: optuna.Trial) -> float:
        maes = []
        for i, fold in enumerate(folds):
            maes += _score(make, defaults, space(trial), feats, cols, [fold])
            trial.report(float(np.mean(maes)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials)
    default_mae = float(np.mean(_score(make, defaults, None, feats, cols, folds)))
    return {
        "params": study.best_params,
        "cv_mae": float(study.best_value),
        "default_cv_mae": default_mae,
        "trials": len(study.trials),
        "completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
    }
