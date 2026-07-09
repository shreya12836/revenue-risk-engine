"""Grouped-bar model comparison chart, replacing the raw metrics table.

Reads the same ``metrics.json`` dict Page 1 already loads (keys: baseline,
xgboost_default, xgboost_tuned), so no new data source is needed.
"""

from __future__ import annotations

import plotly.graph_objects as go

from components.theme import apply_theme

COMPARISON_METRICS = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}
MODEL_LABELS = {
    "baseline": "Baseline (logistic regression)",
    "xgboost_default": "XGBoost (default params)",
    "xgboost_tuned": "XGBoost (tuned)",
}


def model_comparison_figure(metrics: dict[str, dict]) -> go.Figure:
    """Grouped bar chart of headline metrics across every model in ``metrics``."""
    fig = go.Figure()
    for model_key, model_metrics in metrics.items():
        values = [model_metrics.get(m) for m in COMPARISON_METRICS]
        fig.add_trace(
            go.Bar(
                name=MODEL_LABELS.get(model_key, model_key),
                x=[METRIC_LABELS[m] for m in COMPARISON_METRICS],
                y=values,
                hovertemplate="%{x}: %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        title="Model comparison",
        barmode="group",
        yaxis={"title": "Score", "range": [0, 1]},
    )
    return apply_theme(fig)
