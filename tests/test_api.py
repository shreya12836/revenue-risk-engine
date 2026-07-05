"""Tests for the FastAPI inference service.

CI checks out a fresh clone with no trained artifacts (outputs/ is
gitignored), so every test here builds its own tiny fitted model +
feature_schema.json in tmp_path rather than depending on a real
outputs/<timestamp>/ folder.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api import dependencies
from api.dependencies import get_predictor, resolve_artifacts_dir
from api.main import app
from api.schemas import (
    FEATURE_COLUMNS,
    MAX_BATCH_SIZE,
    NULLABLE_FEATURE_COLUMNS,
    PredictionRequest,
)
from models.predict import ChurnPredictor, build_feature_schema, save_feature_schema


@pytest.fixture(autouse=True)
def _reset_predictor_state():
    get_predictor.cache_clear()
    yield
    app.dependency_overrides.clear()
    get_predictor.cache_clear()


@pytest.fixture
def api_artifacts(tmp_path):
    """Tiny real ChurnPredictor artifacts dir, same shape as production."""
    rng = np.random.RandomState(0)
    n = 40
    X = pd.DataFrame({col: rng.uniform(1, 100, n) for col in FEATURE_COLUMNS})
    for col in NULLABLE_FEATURE_COLUMNS:
        X.loc[: n // 4, col] = np.nan  # real low-activity customers have NaN here
    y = pd.Series((X["recency_days"] > 50).astype(int).values)

    model = XGBClassifier(
        n_estimators=10, max_depth=2, eval_metric="logloss", random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, tmp_path / "model_v1.joblib")
    save_feature_schema(build_feature_schema(X), tmp_path / "feature_schema.json")
    return tmp_path


@pytest.fixture
def predictor(api_artifacts):
    return ChurnPredictor.from_artifacts(api_artifacts, version="v1")


@pytest.fixture
def client(predictor):
    app.dependency_overrides[get_predictor] = lambda: predictor
    return TestClient(app)


def _feature_value(name: str) -> int | float | None:
    if name in NULLABLE_FEATURE_COLUMNS:
        return None
    annotation = PredictionRequest.model_fields[name].annotation
    return 5 if annotation is int else 12.5


@pytest.fixture
def valid_payload():
    payload: dict = {name: _feature_value(name) for name in FEATURE_COLUMNS}
    payload["customer_id"] = "CUST-001"
    return payload


class TestHealth:
    def test_returns_ok_status_and_model_version(self, client, predictor):
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_version"] == predictor.schema["version"]


class TestPredictionRequestSchema:
    def test_fields_match_feature_columns_exactly(self):
        field_names = set(PredictionRequest.model_fields) - {"customer_id"}

        assert field_names == set(FEATURE_COLUMNS)


class TestPredictEndpoint:
    def test_valid_request_returns_probability_matching_direct_call(
        self, client, predictor, valid_payload
    ):
        response = client.post("/predict", json=valid_payload)

        assert response.status_code == 200
        row = {k: v for k, v in valid_payload.items() if k != "customer_id"}
        expected = predictor.predict_proba(pd.DataFrame([row], dtype=float))[0]
        body = response.json()
        assert body["churn_probability"] == pytest.approx(float(expected))
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_customer_id_is_echoed_back(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)

        assert response.json()["customer_id"] == "CUST-001"

    def test_customer_id_omitted_echoes_as_none(self, client, valid_payload):
        del valid_payload["customer_id"]

        response = client.post("/predict", json=valid_payload)

        assert response.json()["customer_id"] is None

    def test_missing_required_field_returns_422(self, client, valid_payload):
        del valid_payload["monetary"]

        response = client.post("/predict", json=valid_payload)

        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client, valid_payload):
        valid_payload["monetary"] = "not-a-number"

        response = client.post("/predict", json=valid_payload)

        assert response.status_code == 422

    def test_unknown_extra_field_returns_422(self, client, valid_payload):
        valid_payload["not_a_real_feature"] = 1.0

        response = client.post("/predict", json=valid_payload)

        assert response.status_code == 422

    def test_negative_value_on_nonnegative_field_returns_422(
        self, client, valid_payload
    ):
        valid_payload["recency_days"] = -1

        response = client.post("/predict", json=valid_payload)

        assert response.status_code == 422

    def test_nullable_field_omitted_matches_explicit_null(self, client, valid_payload):
        omitted = dict(valid_payload)
        del omitted["spend_slope"]
        explicit_null = dict(valid_payload)
        explicit_null["spend_slope"] = None

        response_omitted = client.post("/predict", json=omitted)
        response_null = client.post("/predict", json=explicit_null)

        assert response_omitted.status_code == response_null.status_code == 200
        assert (
            response_omitted.json()["churn_probability"]
            == response_null.json()["churn_probability"]
        )


class TestPredictBatchEndpoint:
    def test_batch_returns_predictions_in_order(self, client, valid_payload):
        second = dict(valid_payload)
        second["customer_id"] = "CUST-002"
        second["monetary"] = 999.0

        response = client.post(
            "/predict/batch", json={"records": [valid_payload, second]}
        )

        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 2
        assert predictions[0]["customer_id"] == "CUST-001"
        assert predictions[1]["customer_id"] == "CUST-002"

    def test_empty_batch_returns_422(self, client):
        response = client.post("/predict/batch", json={"records": []})

        assert response.status_code == 422

    def test_batch_over_max_size_returns_422(self, client, valid_payload):
        records = [valid_payload] * (MAX_BATCH_SIZE + 1)

        response = client.post("/predict/batch", json={"records": records})

        assert response.status_code == 422

    def test_one_invalid_record_rejects_whole_batch(self, client, valid_payload):
        invalid = dict(valid_payload)
        invalid["monetary"] = "bad"

        response = client.post(
            "/predict/batch", json={"records": [valid_payload, invalid]}
        )

        assert response.status_code == 422


class TestGlobalExceptionHandling:
    def test_value_error_from_predictor_returns_400(self, client, valid_payload):
        class _RaisingPredictor:
            def predict_proba(self, X):
                raise ValueError("bad column(s)")

        app.dependency_overrides[get_predictor] = lambda: _RaisingPredictor()

        response = client.post("/predict", json=valid_payload)

        assert response.status_code == 400
        assert "bad column(s)" in response.json()["detail"]

    def test_unexpected_error_returns_generic_500(self, client, valid_payload):
        class _BrokenPredictor:
            def predict_proba(self, X):
                raise RuntimeError("super secret internal detail")

        app.dependency_overrides[get_predictor] = lambda: _BrokenPredictor()
        # Starlette's TestClient re-raises unhandled server exceptions by
        # default (useful for most tests); disable that here since we're
        # deliberately verifying the catch-all handler's HTTP response.
        no_raise_client = TestClient(app, raise_server_exceptions=False)

        response = no_raise_client.post("/predict", json=valid_payload)

        assert response.status_code == 500
        assert "super secret internal detail" not in response.text
        assert response.json() == {"detail": "Internal server error."}

    def test_docs_endpoint_available(self, client):
        response = client.get("/docs")

        assert response.status_code == 200


class TestStartupLifespan:
    def test_boots_successfully_with_valid_artifacts(self, api_artifacts, monkeypatch):
        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(api_artifacts))

        with TestClient(app) as boot_client:
            response = boot_client.get("/health")

        assert response.status_code == 200

    def test_fails_fast_when_artifacts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(tmp_path))

        with pytest.raises(FileNotFoundError):
            with TestClient(app):
                pass


class TestResolveArtifactsDir:
    def test_env_var_override_used_when_set(self, api_artifacts, monkeypatch):
        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(api_artifacts))

        assert resolve_artifacts_dir() == api_artifacts

    def test_env_var_override_raises_if_model_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(tmp_path))

        with pytest.raises(FileNotFoundError):
            resolve_artifacts_dir()

    def test_auto_discovers_most_recently_modified_candidate(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("MODEL_ARTIFACTS_DIR", raising=False)
        monkeypatch.setattr(dependencies, "ROOT", tmp_path)

        older = tmp_path / "outputs" / "20260101T000000Z"
        newer = tmp_path / "outputs" / "20260102T000000Z"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)
        (older / "model_v1.joblib").write_bytes(b"old")
        (newer / "model_v1.joblib").write_bytes(b"new")
        os.utime(older / "model_v1.joblib", (1, 1))
        os.utime(newer / "model_v1.joblib", (2, 2))

        assert resolve_artifacts_dir() == newer

    def test_raises_when_no_candidates_exist(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MODEL_ARTIFACTS_DIR", raising=False)
        monkeypatch.setattr(dependencies, "ROOT", tmp_path)
        (tmp_path / "outputs").mkdir()

        with pytest.raises(FileNotFoundError):
            resolve_artifacts_dir()
