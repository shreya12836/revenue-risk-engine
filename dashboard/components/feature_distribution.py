"""Feature-distribution comparison: one customer's value vs. the population."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.theme import RISK_HIGH, apply_theme


def feature_distribution_figure(
    population_df: pd.DataFrame,
    feature: str,
    customer_value: float,
    feature_label: str | None = None,
) -> go.Figure:
    """Histogram of ``feature`` across ``population_df`` with a vline at ``customer_value``."""
    label = feature_label or feature
    fig = px.histogram(
        population_df,
        x=feature,
        nbins=30,
        title=f"{label}: this customer vs. the current population",
        labels={feature: label},
    )
    fig.add_vline(
        x=customer_value,
        line_dash="dash",
        line_color=RISK_HIGH,
        annotation_text="This customer",
    )
    return apply_theme(fig)
