"""Tests for the Optuna hyperparameter search (Day 6).

Mirrors ``test_models.py``'s framing: correctness on small deterministic
data, no mutation of inputs, and an explicit structural check that the
search objective optimizes PR-AUC (not ROC-AUC) since churn is imbalanced.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

import models.tuning as tuning_module
from models.tuning import _cv_pr_auc, train_xgboost_with_params, tune_xgboost

optuna.logging.set_verbosity(optuna.logging.WARNING)

_PARAMS = {
    "n_estimators": 50,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


@pytest.fixture
def tuning_fixtures():
    """A deterministic 60-row dataset, large enough for TimeSeriesSplit(3)."""
    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame(
        {
            "recency_days": rng.uniform(0, 100, n),
            "monetary": rng.uniform(10, 1000, n),
            "frequency": rng.uniform(1, 20, n),
        }
    )
    y = pd.Series((X["recency_days"] > 50).astype(int).values)
    return X, y


class TestCvPrAuc:
    def test_returns_score_in_valid_range(self, tuning_fixtures):
        X, y = tuning_fixtures
        score = _cv_pr_auc(X, y, _PARAMS, n_folds=3, random_state=42)
        assert 0.0 <= score <= 1.0

    def test_never_imports_roc_auc_score(self):
        # Structural guarantee that the tuning objective cannot silently
        # regress to optimizing ROC-AUC, which is optimistic under the
        # class imbalance present in churn labels.
        assert not hasattr(tuning_module, "roc_auc_score")

    def test_uses_average_precision_score(self, tuning_fixtures, monkeypatch):
        X, y = tuning_fixtures
        calls = {"count": 0}
        original = tuning_module.average_precision_score

        def spy(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(tuning_module, "average_precision_score", spy)
        _cv_pr_auc(X, y, _PARAMS, n_folds=3, random_state=42)

        assert calls["count"] > 0


class TestTuneXgboost:
    def test_returns_study_with_expected_param_keys(self, tuning_fixtures):
        X, y = tuning_fixtures
        study = tune_xgboost(X, y, n_trials=3, timeout=60, n_folds=3)
        assert set(_PARAMS.keys()).issubset(study.best_params.keys())

    def test_respects_n_trials_budget(self, tuning_fixtures):
        X, y = tuning_fixtures
        study = tune_xgboost(X, y, n_trials=3, timeout=60, n_folds=3)
        assert len(study.trials) == 3

    def test_best_value_is_valid_pr_auc(self, tuning_fixtures):
        X, y = tuning_fixtures
        study = tune_xgboost(X, y, n_trials=3, timeout=60, n_folds=3)
        assert 0.0 <= study.best_value <= 1.0

    def test_is_reproducible_with_same_random_state(self, tuning_fixtures):
        X, y = tuning_fixtures
        study_a = tune_xgboost(X, y, n_trials=3, timeout=60, n_folds=3, random_state=7)
        study_b = tune_xgboost(X, y, n_trials=3, timeout=60, n_folds=3, random_state=7)
        assert study_a.best_params == study_b.best_params


class TestTrainXgboostWithParams:
    def test_fits_and_predicts_valid_probabilities(self, tuning_fixtures):
        X, y = tuning_fixtures
        model = train_xgboost_with_params(X, y, _PARAMS)
        proba = model.predict_proba(X)[:, 1]
        assert proba.shape == (len(X),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_does_not_mutate_training_data(self, tuning_fixtures):
        X, y = tuning_fixtures
        original_X = X.copy()
        original_y = y.copy()

        train_xgboost_with_params(X, y, _PARAMS)

        pd.testing.assert_frame_equal(X, original_X)
        pd.testing.assert_series_equal(y, original_y)
