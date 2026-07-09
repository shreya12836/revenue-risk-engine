"""Tests for dashboard.services.explainability's pure aggregation/text logic.

Uses lightweight stand-ins for ``shap.Explanation`` (plain objects exposing
``.values``/``.data``/``.feature_names``/``.base_values``) since these
functions only rely on attribute access, not any SHAP-library behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services.explainability import (
    PopulationExplanation,
    business_readable_explanation,
    humanize_feature_name,
    top_churn_drivers,
)


@dataclass
class _FakeBatchExplanation:
    values: np.ndarray
    feature_names: list[str]


class TestHumanizeFeatureName:
    def test_snake_case_with_digits(self):
        assert humanize_feature_name("spend_90d") == "Spend 90d"

    def test_simple_snake_case(self):
        assert humanize_feature_name("recency_days") == "Recency Days"


class TestTopChurnDrivers:
    def test_ranks_features_by_mean_abs_shap_within_mask(self):
        values = np.array(
            [
                [0.5, 0.1],
                [0.6, -0.05],
                [-0.1, 0.9],
            ]
        )
        shap_values = _FakeBatchExplanation(
            values=values, feature_names=["recency_days", "frequency"]
        )
        customers = pd.DataFrame({"churn_probability": [0.9, 0.8, 0.1]})
        pop = PopulationExplanation(shap_values=shap_values, customers=customers, artifacts_dir="x")
        mask = customers["churn_probability"] >= 0.5

        drivers = top_churn_drivers(pop, mask, top_n=2)

        assert list(drivers["feature"]) == ["recency_days", "frequency"]
        assert drivers["mean_abs_shap"].iloc[0] == pytest.approx(0.55)

    def test_empty_mask_returns_empty_dataframe(self):
        values = np.array([[0.5, 0.1]])
        shap_values = _FakeBatchExplanation(
            values=values, feature_names=["recency_days", "frequency"]
        )
        customers = pd.DataFrame({"churn_probability": [0.1]})
        pop = PopulationExplanation(shap_values=shap_values, customers=customers, artifacts_dir="x")
        mask = customers["churn_probability"] >= 0.9

        drivers = top_churn_drivers(pop, mask)

        assert drivers.empty


class TestBusinessReadableExplanation:
    def test_produces_one_sentence_per_top_feature(self):
        shap_row = SimpleNamespace(
            values=np.array([0.8, -0.3, 0.05]),
            data=np.array([12.0, 200.0, 3.0]),
            feature_names=["recency_days", "spend_90d", "frequency"],
            base_values=np.array([0.2]),
        )

        sentences = business_readable_explanation(shap_row, top_k=2)

        assert len(sentences) == 2
        assert "Recency Days" in sentences[0]
        assert "raising" in sentences[0]
        assert "12" in sentences[0]

    def test_negative_shap_value_reads_as_lowering(self):
        shap_row = SimpleNamespace(
            values=np.array([-0.8]),
            data=np.array([5.0]),
            feature_names=["frequency"],
            base_values=np.array([0.2]),
        )

        sentences = business_readable_explanation(shap_row, top_k=1)

        assert "lowering" in sentences[0]
