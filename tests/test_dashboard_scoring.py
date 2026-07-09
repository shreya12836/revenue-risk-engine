"""Tests for dashboard.services.scoring's pure functions and single_customer validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from xgboost import XGBClassifier

from api.schemas import FEATURE_COLUMNS, NULLABLE_FEATURE_COLUMNS, PredictionRequest
from models.predict import ChurnPredictor, build_feature_schema
from services.scoring import attach_revenue_at_risk, score_population
from services.single_customer import build_prediction_request, explain_single_customer


def _feature_value(name: str) -> int | float | None:
    """Mirrors tests/test_api.py's helper: int-typed fields need a whole number."""
    if name in NULLABLE_FEATURE_COLUMNS:
        return None
    annotation = PredictionRequest.model_fields[name].annotation
    return 5 if annotation is int else 12.5


class TestAttachRevenueAtRisk:
    def test_adds_churn_probability_and_revenue_at_risk_columns(self):
        features_df = pd.DataFrame(
            {"customer_id": [1, 2], "spend_90d": [100.0, 200.0]}
        )
        churn_proba = np.array([0.5, 0.25])

        scored = attach_revenue_at_risk(features_df, churn_proba)

        assert list(scored["churn_probability"]) == [0.5, 0.25]
        assert list(scored["revenue_at_risk"]) == [50.0, 50.0]

    def test_fills_missing_spend_90d_with_zero(self):
        features_df = pd.DataFrame(
            {"customer_id": [1, 2], "spend_90d": [np.nan, 200.0]}
        )
        churn_proba = np.array([0.9, 0.25])

        scored = attach_revenue_at_risk(features_df, churn_proba)

        assert scored["revenue_at_risk"].iloc[0] == 0.0
        assert scored["revenue_at_risk"].iloc[1] == 50.0


class TestScorePopulation:
    @pytest.fixture
    def fitted_predictor(self):
        rng = np.random.RandomState(0)
        n = 30
        X = pd.DataFrame({col: rng.uniform(1, 100, n) for col in FEATURE_COLUMNS})
        y = pd.Series((X["recency_days"] > 50).astype(int).values)
        model = XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss", random_state=42)
        model.fit(X, y)
        schema = build_feature_schema(X)
        return ChurnPredictor(model=model, schema=schema), X

    def test_matches_direct_predict_proba_call(self, fitted_predictor):
        predictor, X = fitted_predictor
        features_df = X.copy()
        features_df.insert(0, "customer_id", range(len(X)))

        result = score_population(predictor, features_df)

        expected = predictor.predict_proba(X)
        np.testing.assert_allclose(result, expected)


class TestExplainSingleCustomer:
    def test_returns_none_and_does_not_raise_when_predict_with_shap_fails(self, monkeypatch):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({col: [rng.uniform(1, 100)] for col in FEATURE_COLUMNS})
        model = XGBClassifier(n_estimators=5, max_depth=2, eval_metric="logloss", random_state=42)
        model.fit(X, pd.Series([0]))
        schema = build_feature_schema(X)
        predictor = ChurnPredictor(model=model, schema=schema)

        def _raise(self, X):
            raise RuntimeError("SHAP explainer blew up")

        monkeypatch.setattr(ChurnPredictor, "predict_with_shap", _raise)

        form_values = {name: _feature_value(name) for name in FEATURE_COLUMNS}
        request = build_prediction_request(form_values, None)

        result = explain_single_customer(predictor, request)

        assert result is None


class TestBuildPredictionRequest:
    def test_valid_dict_returns_prediction_request(self):
        form_values = {name: _feature_value(name) for name in FEATURE_COLUMNS}

        request = build_prediction_request(form_values, "CUST-001")

        assert request.customer_id == "CUST-001"
        assert request.monetary == 12.5

    def test_negative_value_raises_validation_error(self):
        form_values = {name: _feature_value(name) for name in FEATURE_COLUMNS}
        form_values["frequency"] = -1

        with pytest.raises(ValidationError):
            build_prediction_request(form_values, None)

    def test_extra_field_raises_validation_error(self):
        form_values = {name: _feature_value(name) for name in FEATURE_COLUMNS}
        form_values["not_a_real_feature"] = 1.0

        with pytest.raises(ValidationError):
            build_prediction_request(form_values, None)
