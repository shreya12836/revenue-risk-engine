"""Dynamic ROC and Precision-Recall curve figures.

Built from live-scored test-set arrays (``services.evaluation.evaluate_test_set``)
rather than the pre-baked PNGs training saves -- these are pure
``data -> go.Figure`` builders with no ``st.*`` calls, so they're unit
testable without a Streamlit session (same pattern as
``components.confusion_matrix``).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from components.theme import ACCENT, REFERENCE_LINE, apply_theme


def roc_curve_figure(y_true: np.ndarray, y_proba: np.ndarray) -> go.Figure:
    """ROC curve with per-point threshold hover and a random-classifier diagonal."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Tuned XGBoost (AUC = {auc:.3f})",
            line={"color": ACCENT, "width": 2.5},
            customdata=thresholds,
            hovertemplate=(
                "FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>threshold: %{customdata:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random classifier",
            line={"color": REFERENCE_LINE, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.02]},
    )
    return apply_theme(fig)


def pr_curve_figure(y_true: np.ndarray, y_proba: np.ndarray) -> go.Figure:
    """Precision-recall curve with per-point threshold hover and a prevalence baseline."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    prevalence = float(np.asarray(y_true).mean())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"Tuned XGBoost (PR-AUC = {pr_auc:.3f})",
            line={"color": ACCENT, "width": 2.5},
            customdata=np.append(thresholds, np.nan),
            hovertemplate=(
                "Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>"
                "threshold: %{customdata:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[prevalence, prevalence],
            mode="lines",
            name=f"Baseline (prevalence = {prevalence:.3f})",
            line={"color": REFERENCE_LINE, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.02]},
    )
    return apply_theme(fig)
