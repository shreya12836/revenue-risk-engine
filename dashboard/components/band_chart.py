"""Bar chart for a risk-band summary (spend/recency/frequency/tenure bands)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from components.theme import ACCENT, RISK_HIGH, apply_theme


def band_risk_figure(summary_df: pd.DataFrame, title: str) -> go.Figure:
    """Grouped bars: mean churn probability (left axis) + total revenue at risk (right axis)."""
    if summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} -- no data available")
        return apply_theme(fig)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary_df["band"],
            y=summary_df["mean_churn_probability"],
            name="Mean churn probability",
            marker_color=RISK_HIGH,
            hovertemplate="%{x}<br>mean churn probability: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary_df["band"],
            y=summary_df["revenue_at_risk"],
            name="Total revenue at risk",
            mode="lines+markers",
            marker_color=ACCENT,
            yaxis="y2",
            hovertemplate="%{x}<br>revenue at risk: £%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Band",
        yaxis={"title": "Mean churn probability", "range": [0, 1]},
        yaxis2={"title": "Revenue at risk (£)", "overlaying": "y", "side": "right"},
    )
    return apply_theme(fig)
