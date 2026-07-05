"""Tests for SHAP explainability on the tuned XGBoost model (Day 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from models.explain import (
    build_feature_importance_table,
    compute_shap_values,
    save_shap_summary_plot,
    save_shap_waterfall_plot,
)


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
    customer_ids = pd.Series(range(1000, 1000 + n))
    return model, X, customer_ids


class TestComputeShapValues:
    def test_shape_matches_features(self, fitted_model_and_data):
        model, X, _ = fitted_model_and_data
        shap_values = compute_shap_values(model, X)
        assert shap_values.values.shape == (len(X), X.shape[1])


class TestSaveShapSummaryPlot:
    def test_writes_png_file(self, tmp_path, fitted_model_and_data):
        model, X, _ = fitted_model_and_data
        shap_values = compute_shap_values(model, X)
        output_path = tmp_path / "shap_summary.png"

        result = save_shap_summary_plot(shap_values, X, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestSaveShapWaterfallPlot:
    def test_writes_png_file_for_valid_customer(self, tmp_path, fitted_model_and_data):
        model, X, customer_ids = fitted_model_and_data
        shap_values = compute_shap_values(model, X)
        output_path = tmp_path / "shap_waterfall_1005.png"

        result = save_shap_waterfall_plot(
            shap_values, X, customer_ids, customer_id=1005, output_path=output_path
        )

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_raises_for_unknown_customer_id(self, tmp_path, fitted_model_and_data):
        model, X, customer_ids = fitted_model_and_data
        shap_values = compute_shap_values(model, X)

        with pytest.raises(ValueError, match="999999"):
            save_shap_waterfall_plot(
                shap_values,
                X,
                customer_ids,
                customer_id=999999,
                output_path=tmp_path / "out.png",
            )


class TestBuildFeatureImportanceTable:
    def test_has_expected_columns(self, fitted_model_and_data):
        model, X, _ = fitted_model_and_data
        shap_values = compute_shap_values(model, X)
        table = build_feature_importance_table(model, shap_values, list(X.columns))
        assert set(table.columns) == {"feature", "xgboost_importance", "mean_abs_shap"}

    def test_row_count_equals_feature_count(self, fitted_model_and_data):
        model, X, _ = fitted_model_and_data
        shap_values = compute_shap_values(model, X)
        table = build_feature_importance_table(model, shap_values, list(X.columns))
        assert len(table) == X.shape[1]

    def test_sorted_by_mean_abs_shap_descending(self, fitted_model_and_data):
        model, X, _ = fitted_model_and_data
        shap_values = compute_shap_values(model, X)
        table = build_feature_importance_table(model, shap_values, list(X.columns))
        values = table["mean_abs_shap"].tolist()
        assert values == sorted(values, reverse=True)

    def test_feature_names_match_input_columns(self, fitted_model_and_data):
        model, X, _ = fitted_model_and_data
        shap_values = compute_shap_values(model, X)
        table = build_feature_importance_table(model, shap_values, list(X.columns))
        assert set(table["feature"]) == set(X.columns)
