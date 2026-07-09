"""Interactive SHAP waterfall for a single customer's prediction.

``shap>=0.42`` has no native Plotly waterfall (only a matplotlib one via
``shap.plots.waterfall``), so this builds a ``go.Waterfall`` directly from
the ``shap.Explanation``'s raw arrays (``.values``/``.base_values``/``.data``/
``.feature_names``) -- no embedded image, fully interactive (hover, zoom).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import shap

from components.theme import RISK_HIGH, RISK_LOW, apply_theme


def shap_waterfall_figure(shap_row: shap.Explanation, top_k: int = 12) -> go.Figure:
    """Build a waterfall chart for one customer's SHAP explanation.

    ``shap_row`` must be a single-row ``shap.Explanation`` (e.g.
    ``shap_values[0]`` from a batch of one). Shows the ``top_k`` features by
    |SHAP value|; the remainder are aggregated into one "other features" bar
    so the chart stays readable for a 33-feature model.
    """
    values = np.asarray(shap_row.values)
    data = np.asarray(shap_row.data)
    feature_names = list(shap_row.feature_names)
    base_value = float(np.asarray(shap_row.base_values).reshape(-1)[0])

    order = np.argsort(-np.abs(values))
    top_idx = order[:top_k]
    rest_idx = order[top_k:]

    labels = [f"{feature_names[i]} = {data[i]:g}" for i in top_idx]
    contributions = [float(values[i]) for i in top_idx]
    if len(rest_idx):
        labels.append(f"{len(rest_idx)} other features")
        contributions.append(float(values[rest_idx].sum()))

    x_labels = ["Base value", *labels, "Prediction"]
    y_values = [base_value, *contributions, 0.0]
    measures = ["absolute", *(["relative"] * len(contributions)), "total"]

    fig = go.Figure(
        go.Waterfall(
            x=x_labels,
            y=y_values,
            measure=measures,
            increasing={"marker": {"color": RISK_HIGH}},
            decreasing={"marker": {"color": RISK_LOW}},
            totals={"marker": {"color": "#64748B"}},
            connector={"line": {"color": "#CBD5E1"}},
            hovertemplate="%{x}<br>contribution: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Why this prediction -- SHAP contributions",
        yaxis_title="Contribution to churn probability (log-odds)",
        showlegend=False,
    )
    return apply_theme(fig)
