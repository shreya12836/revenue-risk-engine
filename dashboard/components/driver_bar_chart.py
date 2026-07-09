"""Top churn drivers across a filtered customer population -- horizontal bar."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.theme import RISK_HIGH, apply_theme


def driver_bar_figure(drivers_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of ``drivers_df`` (columns: feature, mean_abs_shap)."""
    if drivers_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No customers match the current filter")
        return apply_theme(fig)

    plot_df = drivers_df.sort_values("mean_abs_shap")
    fig = px.bar(
        plot_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Top churn drivers for the selected customers",
        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
        color_discrete_sequence=[RISK_HIGH],
    )
    return apply_theme(fig)
