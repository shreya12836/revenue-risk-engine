"""Tests for ChurnPredictor: schema-validated inference (Day 6).

``feature_schema.json`` exists specifically to back ``predict_proba``'s
boundary validation — a DataFrame payload can't be validated by Pydantic
alone, so this schema is the equivalent check for tabular input.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from models.predictor import ChurnPredictor, build_feature_schema, save_feature_schema


@pytest.fixture
def fitted_model_and_data():
    rng = np.random.RandomState(0)
    n = 30
    X = pd.DataFrame(
        {
            "recency_days": rng.uniform(0, 100, n),
            "monetary": rng.uniform(10, 1000, n),
            "frequency": rng.uniform(1, 20, n),
        }
    )
    y = pd.Series((X["recency_days"] > 50).astype(int).values)
    model = XGBClassifier(
        n_estimators=10, max_depth=2, eval_metric="logloss", random_state=42
    )
    model.fit(X, y)
    return model, X


class TestBuildFeatureSchema:
    def test_contains_all_feature_columns(self, fitted_model_and_data):
        _, X = fitted_model_and_data
        schema = build_feature_schema(X)
        assert schema["feature_columns"] == list(X.columns)

    def test_default_version_is_v1(self, fitted_model_and_data):
        _, X = fitted_model_and_data
        schema = build_feature_schema(X)
        assert schema["version"] == "v1"

    def test_dtypes_recorded_per_column(self, fitted_model_and_data):
        _, X = fitted_model_and_data
        schema = build_feature_schema(X)
        assert set(schema["dtypes"].keys()) == set(X.columns)


class TestSaveFeatureSchema:
    def test_writes_json_file(self, tmp_path, fitted_model_and_data):
        _, X = fitted_model_and_data
        schema = build_feature_schema(X)
        output_path = tmp_path / "feature_schema.json"

        result = save_feature_schema(schema, output_path)

        assert result == output_path
        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["feature_columns"] == list(X.columns)


class TestChurnPredictor:
    def test_predict_proba_matches_direct_model_call(self, fitted_model_and_data):
        model, X = fitted_model_and_data
        schema = build_feature_schema(X)
        predictor = ChurnPredictor(model=model, schema=schema)

        expected = model.predict_proba(X)[:, 1]
        actual = predictor.predict_proba(X)

        np.testing.assert_allclose(actual, expected)

    def test_predict_proba_reorders_columns_to_schema_order(self, fitted_model_and_data):
        model, X = fitted_model_and_data
        schema = build_feature_schema(X)
        predictor = ChurnPredictor(model=model, schema=schema)

        shuffled = X[list(reversed(X.columns))]
        expected = model.predict_proba(X)[:, 1]
        actual = predictor.predict_proba(shuffled)

        np.testing.assert_allclose(actual, expected)

    def test_raises_on_missing_feature_column(self, fitted_model_and_data):
        model, X = fitted_model_and_data
        schema = build_feature_schema(X)
        predictor = ChurnPredictor(model=model, schema=schema)

        incomplete = X.drop(columns=["monetary"])
        with pytest.raises(ValueError, match="monetary"):
            predictor.predict_proba(incomplete)

    def test_raises_on_unexpected_extra_column(self, fitted_model_and_data):
        model, X = fitted_model_and_data
        schema = build_feature_schema(X)
        predictor = ChurnPredictor(model=model, schema=schema)

        extra = X.copy()
        extra["unexpected_col"] = 1.0
        with pytest.raises(ValueError, match="unexpected_col"):
            predictor.predict_proba(extra)

    def test_from_artifacts_round_trips(self, tmp_path, fitted_model_and_data):
        model, X = fitted_model_and_data
        schema = build_feature_schema(X)
        joblib.dump(model, tmp_path / "model_v1.joblib")
        save_feature_schema(schema, tmp_path / "feature_schema.json")

        predictor = ChurnPredictor.from_artifacts(tmp_path, version="v1")

        expected = model.predict_proba(X)[:, 1]
        actual = predictor.predict_proba(X)
        np.testing.assert_allclose(actual, expected)
