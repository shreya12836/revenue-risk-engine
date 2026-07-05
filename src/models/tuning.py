"""Optuna hyperparameter search for XGBoost (Day 6).

The search objective is **PR-AUC** (``average_precision_score``), not
ROC-AUC. Churn is imbalanced, and ROC-AUC is optimistic under imbalance —
precision-recall trade-offs are what actually matter for a churn model
whose predictions drive limited retention spend.

The CV loop mirrors ``cv.run_time_cv``'s ``TimeSeriesSplit`` + per-fold
``scale_pos_weight`` pattern, but is XGBoost-only and parameterized by
trial-suggested hyperparameters instead of fixed defaults.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from utils.logger import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42


def _suggest_params(trial: optuna.Trial) -> dict:
    """Sample one XGBoost hyperparameter set for a trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _cv_pr_auc(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_folds: int,
    random_state: int,
) -> float:
    """Mean PR-AUC across ``TimeSeriesSplit`` folds for one hyperparameter set.

    Folds whose training portion has only one class are skipped (mirrors
    ``run_time_cv``'s handling of the same edge case in tiny CV folds).
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)
    scores: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        if y_train_fold.nunique() < 2:
            continue

        neg = int((y_train_fold == 0).sum())
        pos = int((y_train_fold == 1).sum())
        scale_pos_weight = neg / pos if pos else 1.0

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
        )
        model.fit(X_train_fold, y_train_fold)
        proba = model.predict_proba(X_val_fold)[:, 1]
        scores.append(average_precision_score(y_val_fold, proba))

    if not scores:
        raise ValueError("No CV fold had both classes in its training split")
    return float(np.mean(scores))


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    timeout: int,
    n_folds: int = 5,
    random_state: int = RANDOM_STATE,
) -> optuna.Study:
    """Run an Optuna search maximizing mean CV PR-AUC for XGBoost.

    ``n_trials``/``timeout`` are expected to come from
    ``config.modeling.hyperparameter_tuning`` rather than being hardcoded
    by callers.
    """

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        return _cv_pr_auc(X_train, y_train, params, n_folds, random_state)

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(
        "Optuna search finished: best_pr_auc=%.4f best_params=%s",
        study.best_value,
        study.best_params,
    )
    return study


def train_xgboost_with_params(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    """Fit XGBoost on the full data using externally supplied hyperparameters.

    Kept separate from ``train.train_xgboost`` (which always uses fixed
    defaults) so the default-hyperparameter training path stays untouched.
    """
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    scale_pos_weight = neg / pos if pos else 1.0

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X, y)
    return model
