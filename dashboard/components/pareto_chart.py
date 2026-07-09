"""Pareto view: % of revenue at risk captured by the top X% of customers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.theme import ACCENT, REFERENCE_LINE, apply_theme

PARETO_TARGET = 0.8


def pareto_figure(customers_df: pd.DataFrame) -> go.Figure:
    """Cumulative % of total revenue at risk vs. % of customers (highest risk first).

    Annotates the smallest customer-% needed to capture ``PARETO_TARGET``
    (80%) of total revenue at risk -- computed from the data, not guessed.
    """
    if customers_df.empty or customers_df["revenue_at_risk"].sum() == 0:
        fig = go.Figure()
        fig.update_layout(title="No revenue-at-risk data to plot")
        return apply_theme(fig)

    sorted_revenue = customers_df["revenue_at_risk"].sort_values(ascending=False).to_numpy()
    total = sorted_revenue.sum()
    cumulative_pct = np.cumsum(sorted_revenue) / total
    n = len(sorted_revenue)
    customer_pct = np.arange(1, n + 1) / n * 100

    target_idx = int(np.searchsorted(cumulative_pct, PARETO_TARGET))
    target_idx = min(target_idx, n - 1)
    target_customer_pct = customer_pct[target_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=customer_pct,
            y=cumulative_pct * 100,
            mode="lines",
            name="Cumulative % of revenue at risk",
            line={"color": ACCENT, "width": 2.5},
            hovertemplate=(
                "Top %{x:.1f}% of customers<br>"
                "captures %{y:.1f}% of revenue at risk<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=PARETO_TARGET * 100,
        line_dash="dash",
        line_color=REFERENCE_LINE,
        annotation_text=f"{PARETO_TARGET:.0%} of revenue at risk",
    )
    fig.add_vline(
        x=target_customer_pct,
        line_dash="dash",
        line_color=REFERENCE_LINE,
        annotation_text=f"Top {target_customer_pct:.0f}% of customers",
    )
    fig.update_layout(
        title=(
            f"Pareto view -- top {target_customer_pct:.0f}% of customers account for "
            f"{PARETO_TARGET:.0%} of revenue at risk"
        ),
        xaxis_title="% of customers (ranked by revenue at risk)",
        yaxis_title="Cumulative % of revenue at risk",
        xaxis={"range": [0, 100]},
        yaxis={"range": [0, 102]},
    )
    return apply_theme(fig)
