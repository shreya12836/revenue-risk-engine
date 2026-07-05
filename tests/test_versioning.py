"""Tests for model versioning: metadata, git commit hash, artifact saving."""

from __future__ import annotations

import json
import subprocess

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
from xgboost import XGBClassifier

from models.versioning import build_metadata, get_git_commit_hash, save_model_version

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def fitted_model_and_study():
    rng = np.random.RandomState(0)
    n = 40
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
    y = pd.Series((X["a"] > 0.5).astype(int).values)
    model = XGBClassifier(
        n_estimators=10, max_depth=2, eval_metric="logloss", random_state=42
    )
    model.fit(X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=2)
    return model, study


class TestGetGitCommitHash:
    def test_returns_nonempty_string_in_repo(self):
        commit = get_git_commit_hash()
        assert isinstance(commit, str)
        assert len(commit) > 0
        assert commit != "unknown"

    def test_falls_back_to_unknown_on_failure(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "git")

        monkeypatch.setattr("models.versioning.subprocess.run", fake_run)
        assert get_git_commit_hash() == "unknown"

    def test_falls_back_to_unknown_when_git_missing(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr("models.versioning.subprocess.run", fake_run)
        assert get_git_commit_hash() == "unknown"


class TestBuildMetadata:
    def test_has_required_keys(self, fitted_model_and_study):
        model, study = fitted_model_and_study
        metadata = build_metadata(
            model,
            feature_count=2,
            split_sizes={"train": 100, "val": 20, "test": 30},
            study=study,
            cv_folds=5,
        )
        required = {
            "version",
            "git_commit",
            "training_date",
            "algorithm",
            "feature_count",
            "train_samples",
            "val_samples",
            "test_samples",
            "cv_folds",
            "optuna_trials",
            "optuna_best_pr_auc",
            "hyperparameters",
        }
        assert required.issubset(metadata.keys())

    def test_feature_count_and_split_sizes_match_input(self, fitted_model_and_study):
        model, study = fitted_model_and_study
        metadata = build_metadata(
            model,
            feature_count=2,
            split_sizes={"train": 100, "val": 20, "test": 30},
            study=study,
            cv_folds=5,
        )
        assert metadata["feature_count"] == 2
        assert metadata["train_samples"] == 100
        assert metadata["val_samples"] == 20
        assert metadata["test_samples"] == 30

    def test_optuna_trials_matches_study(self, fitted_model_and_study):
        model, study = fitted_model_and_study
        metadata = build_metadata(
            model,
            feature_count=2,
            split_sizes={"train": 1, "val": 1, "test": 1},
            study=study,
            cv_folds=5,
        )
        assert metadata["optuna_trials"] == len(study.trials)

    def test_default_version_is_v1(self, fitted_model_and_study):
        model, study = fitted_model_and_study
        metadata = build_metadata(
            model,
            feature_count=2,
            split_sizes={"train": 1, "val": 1, "test": 1},
            study=study,
            cv_folds=5,
        )
        assert metadata["version"] == "v1"


class TestSaveModelVersion:
    def test_writes_model_and_metadata_files(self, tmp_path, fitted_model_and_study):
        model, study = fitted_model_and_study
        metadata = build_metadata(
            model,
            feature_count=2,
            split_sizes={"train": 1, "val": 1, "test": 1},
            study=study,
            cv_folds=5,
        )
        model_path = save_model_version(model, metadata, tmp_path, version="v1")

        assert model_path.exists()
        assert model_path.name == "model_v1.joblib"
        metadata_path = tmp_path / "metadata.json"
        assert metadata_path.exists()
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert loaded["version"] == "v1"

    def test_saved_model_round_trips(self, tmp_path, fitted_model_and_study):
        model, study = fitted_model_and_study
        metadata = build_metadata(
            model,
            feature_count=2,
            split_sizes={"train": 1, "val": 1, "test": 1},
            study=study,
            cv_folds=5,
        )
        model_path = save_model_version(model, metadata, tmp_path, version="v1")
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, "predict_proba")
