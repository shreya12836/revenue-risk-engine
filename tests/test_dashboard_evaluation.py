"""Tests for dashboard.services.evaluation's pure scoring function.

``load_test_split``/``evaluate_test_set`` are not exercised here -- they
require the real dataset file and full feature pipeline, mirroring
``test_dashboard_scoring.py``'s precedent of only unit-testing the pure
``score_population`` step rather than its cached orchestrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from api.schemas import FEATURE_COLUMNS
from models.predict import ChurnPredictor, build_feature_schema
from services.evaluation import score_test_set


class TestScoreTestSet:
    def test_matches_direct_predict_proba_call(self):
        rng = np.random.RandomState(0)
        n = 30
        X = pd.DataFrame({col: rng.uniform(1, 100, n) for col in FEATURE_COLUMNS})
        y = pd.Series((X["recency_days"] > 50).astype(int).values)
        model = XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss", random_state=42)
        model.fit(X, y)
        schema = build_feature_schema(X)
        predictor = ChurnPredictor(model=model, schema=schema)

        result = score_test_set(predictor, X)

        expected = predictor.predict_proba(X)
        np.testing.assert_allclose(result, expected)
