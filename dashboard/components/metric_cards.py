"""Reusable ``st.metric`` row rendering."""

from __future__ import annotations

import streamlit as st

METRIC_LABELS: dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "brier_score": "Brier score",
    "lift_at_10pct": "Lift @ top 10%",
}


def render_metric_row(metrics: dict[str, float], keys: list[str]) -> None:
    """Render one ``st.metric`` per key in ``keys``, laid out in even columns."""
    columns = st.columns(len(keys))
    for column, key in zip(columns, keys):
        with column:
            label = METRIC_LABELS.get(key, key)
            value = metrics.get(key)
            display = f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"
            st.metric(label, display)
