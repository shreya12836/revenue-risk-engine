"""Tests for dashboard.components -- pure figure-building logic only.

Streamlit rendering calls (``st.plotly_chart``, etc.) aren't exercised here;
only the data -> ``go.Figure`` construction, which is what can actually
break silently. Every builder returns a ``plotly.graph_objects.Figure`` (no
matplotlib/embedded-image charts remain in the dashboard).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from components.band_chart import band_risk_figure
from components.calibration_chart import calibration_curve_figure
from components.confusion_matrix import confusion_matrix_figure
from components.cumulative_risk_chart import cumulative_revenue_at_risk_figure
from components.driver_bar_chart import driver_bar_figure
from components.feature_distribution import feature_distribution_figure
from components.lift_chart import lift_gains_figure
from components.model_comparison import model_comparison_figure
from components.pareto_chart import pareto_figure
from components.roc_pr_curves import pr_curve_figure, roc_curve_figure
from components.shap_chart import shap_waterfall_figure


@pytest.fixture
def binary_predictions():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 100)
    y_proba = np.clip(y_true * 0.5 + rng.uniform(0, 0.5, 100), 0, 1)
    return y_true, y_proba


class TestConfusionMatrixFigure:
    def test_returns_a_plotly_figure(self):
        counts = {"tn": 1199, "fp": 948, "fn": 454, "tp": 2452}

        fig = confusion_matrix_figure(counts)

        assert isinstance(fig, go.Figure)

    def test_handles_all_zero_counts_without_raising(self):
        counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

        fig = confusion_matrix_figure(counts)

        assert isinstance(fig, go.Figure)


class TestRocPrCurveFigures:
    def test_roc_curve_figure_returns_plotly_figure(self, binary_predictions):
        y_true, y_proba = binary_predictions

        fig = roc_curve_figure(y_true, y_proba)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # curve + diagonal reference

    def test_pr_curve_figure_returns_plotly_figure(self, binary_predictions):
        y_true, y_proba = binary_predictions

        fig = pr_curve_figure(y_true, y_proba)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # curve + prevalence baseline


class TestCalibrationCurveFigure:
    def test_returns_plotly_figure(self, binary_predictions):
        y_true, y_proba = binary_predictions

        fig = calibration_curve_figure(y_true, y_proba, n_bins=5)

        assert isinstance(fig, go.Figure)

    def test_handles_single_class_without_raising(self):
        y_true = np.zeros(20, dtype=int)
        y_proba = np.linspace(0.01, 0.5, 20)

        fig = calibration_curve_figure(y_true, y_proba, n_bins=5)

        assert isinstance(fig, go.Figure)


class TestLiftGainsFigure:
    def test_returns_plotly_figure_with_dual_axes(self, binary_predictions):
        y_true, y_proba = binary_predictions

        fig = lift_gains_figure(y_true, y_proba)

        assert isinstance(fig, go.Figure)
        assert fig.layout.yaxis2 is not None

    def test_handles_all_zero_labels_without_raising(self):
        y_true = np.zeros(20, dtype=int)
        y_proba = np.linspace(0, 1, 20)

        fig = lift_gains_figure(y_true, y_proba)

        assert isinstance(fig, go.Figure)


class TestModelComparisonFigure:
    def test_returns_one_trace_per_model(self):
        metrics = {
            "baseline": {
                "roc_auc": 0.6,
                "pr_auc": 0.5,
                "precision": 0.4,
                "recall": 0.3,
                "f1": 0.35,
            },
            "xgboost_tuned": {
                "roc_auc": 0.8,
                "pr_auc": 0.75,
                "precision": 0.7,
                "recall": 0.6,
                "f1": 0.65,
            },
        }

        fig = model_comparison_figure(metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


class TestShapWaterfallFigure:
    def test_returns_plotly_figure_from_explanation_row(self):
        shap_row = SimpleNamespace(
            values=np.array([0.5, -0.2, 0.1]),
            data=np.array([10.0, 20.0, 5.0]),
            feature_names=["recency_days", "spend_90d", "frequency"],
            base_values=np.array([0.2]),
        )

        fig = shap_waterfall_figure(shap_row)

        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "waterfall"

    def test_aggregates_features_beyond_top_k(self):
        n_features = 20
        shap_row = SimpleNamespace(
            values=np.linspace(0.01, 1.0, n_features),
            data=np.arange(n_features, dtype=float),
            feature_names=[f"feature_{i}" for i in range(n_features)],
            base_values=np.array([0.3]),
        )

        fig = shap_waterfall_figure(shap_row, top_k=5)

        # base value + 5 top features + "other" bucket + prediction total
        assert len(fig.data[0].x) == 8


class TestCumulativeRevenueAtRiskFigure:
    def test_returns_monotonically_increasing_curve(self):
        customers = pd.DataFrame({"revenue_at_risk": [50.0, 10.0, 30.0]})

        fig = cumulative_revenue_at_risk_figure(customers)

        assert isinstance(fig, go.Figure)
        y = fig.data[0].y
        assert list(y) == sorted(y)

    def test_handles_empty_dataframe_without_raising(self):
        customers = pd.DataFrame(columns=["revenue_at_risk"])

        fig = cumulative_revenue_at_risk_figure(customers)

        assert isinstance(fig, go.Figure)


class TestParetoFigure:
    def test_returns_plotly_figure(self):
        customers = pd.DataFrame({"revenue_at_risk": [100.0, 10.0, 5.0, 1.0]})

        fig = pareto_figure(customers)

        assert isinstance(fig, go.Figure)

    def test_handles_all_zero_revenue_without_raising(self):
        customers = pd.DataFrame({"revenue_at_risk": [0.0, 0.0, 0.0]})

        fig = pareto_figure(customers)

        assert isinstance(fig, go.Figure)


class TestBandRiskFigure:
    def test_returns_plotly_figure(self):
        summary = pd.DataFrame(
            {
                "band": ["low", "high"],
                "count": [10, 5],
                "mean_churn_probability": [0.2, 0.8],
                "revenue_at_risk": [100.0, 500.0],
            }
        )

        fig = band_risk_figure(summary, "Risk by spend band")

        assert isinstance(fig, go.Figure)

    def test_handles_empty_summary_without_raising(self):
        summary = pd.DataFrame(
            columns=["band", "count", "mean_churn_probability", "revenue_at_risk"]
        )

        fig = band_risk_figure(summary, "Risk by spend band")

        assert isinstance(fig, go.Figure)


class TestDriverBarFigure:
    def test_returns_plotly_figure(self):
        drivers = pd.DataFrame(
            {"feature": ["recency_days", "frequency"], "mean_abs_shap": [0.5, 0.3]}
        )

        fig = driver_bar_figure(drivers)

        assert isinstance(fig, go.Figure)

    def test_handles_empty_drivers_without_raising(self):
        drivers = pd.DataFrame(columns=["feature", "mean_abs_shap"])

        fig = driver_bar_figure(drivers)

        assert isinstance(fig, go.Figure)


class TestFeatureDistributionFigure:
    def test_returns_plotly_figure(self):
        population = pd.DataFrame({"recency_days": [1.0, 5.0, 10.0, 20.0]})

        fig = feature_distribution_figure(population, "recency_days", customer_value=7.0)

        assert isinstance(fig, go.Figure)
