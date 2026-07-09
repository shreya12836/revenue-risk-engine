"""Confusion-matrix visualization.

No confusion-matrix image is saved to ``outputs/<timestamp>/figures/`` --
only the raw ``{tn, fp, fn, tp}`` counts in ``metrics.json`` -- so this
renders one from those counts rather than recomputing anything.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from components.theme import apply_theme


def confusion_matrix_figure(counts: dict[str, int]) -> go.Figure:
    """Build a labeled 2x2 heatmap figure from ``{tn, fp, fn, tp}`` counts."""
    matrix = np.array(
        [
            [counts["tn"], counts["fp"]],
            [counts["fn"], counts["tp"]],
        ]
    )
    labels = ["No churn", "Churn"]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False,
            text=matrix,
            texttemplate="%{text:,d}",
            textfont={"size": 16},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>count: %{z:,d}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion matrix (tuned XGBoost)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis={"autorange": "reversed"},
    )
    return apply_theme(fig)
