"""Cumulative revenue-at-risk curve: customers ranked by revenue at risk, descending."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.theme import ACCENT, apply_theme


def cumulative_revenue_at_risk_figure(customers_df: pd.DataFrame) -> go.Figure:
    """Line chart of cumulative revenue at risk vs. customer rank (highest risk first)."""
    if customers_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No customers to plot")
        return apply_theme(fig)

    sorted_revenue = customers_df["revenue_at_risk"].sort_values(ascending=False).to_numpy()
    cumulative = np.cumsum(sorted_revenue)
    rank = np.arange(1, len(cumulative) + 1)

    fig = go.Figure(
        go.Scatter(
            x=rank,
            y=cumulative,
            mode="lines",
            line={"color": ACCENT, "width": 2.5},
            fill="tozeroy",
            hovertemplate="Top %{x:,d} customers<br>cumulative: £%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cumulative revenue at risk",
        xaxis_title="Customers, ranked by revenue at risk (highest first)",
        yaxis_title="Cumulative revenue at risk (£)",
    )
    return apply_theme(fig)
