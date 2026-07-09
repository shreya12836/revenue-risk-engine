"""Lift/cumulative-gains chart, computed fresh from sorted predictions.

``models.evaluate.lift_at_k`` only returns a single scalar (lift at one
fixed k); this builds the full per-decile curve needed for an interactive
chart, using the same "rank by predicted risk, take the top fraction" idea.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from components.theme import ACCENT, REFERENCE_LINE, RISK_HIGH, apply_theme


def lift_gains_figure(y_true: np.ndarray, y_proba: np.ndarray, n_deciles: int = 10) -> go.Figure:
    """Dual-line chart: cumulative % of churners captured, and lift, per decile.

    Customers are ranked by predicted churn probability (highest first) and
    split into ``n_deciles`` equal-sized groups; each point is cumulative
    (e.g. decile 3 = the top 30% of customers by risk).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    order = np.argsort(-y_proba)
    sorted_labels = y_true[order]

    total_positives = sorted_labels.sum()
    base_rate = sorted_labels.mean() if n else 0.0

    deciles = np.arange(1, n_deciles + 1)
    pct_customers = deciles / n_deciles
    cum_gains = []
    lift = []
    for pct in pct_customers:
        cutoff = max(1, int(round(n * pct)))
        captured = sorted_labels[:cutoff].sum()
        gains = (captured / total_positives) if total_positives else 0.0
        cum_gains.append(gains)
        rate = sorted_labels[:cutoff].mean()
        lift.append((rate / base_rate) if base_rate else float("nan"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pct_customers * 100,
            y=np.array(cum_gains) * 100,
            mode="lines+markers",
            name="Cumulative % of churners captured",
            line={"color": ACCENT},
            hovertemplate=(
                "Top %{x:.0f}% of customers<br>captures %{y:.1f}% of churners<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Random targeting",
            line={"color": REFERENCE_LINE, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pct_customers * 100,
            y=lift,
            mode="lines+markers",
            name="Lift",
            line={"color": RISK_HIGH},
            yaxis="y2",
            hovertemplate="Top %{x:.0f}% of customers<br>lift = %{y:.2f}x<extra></extra>",
        )
    )
    fig.update_layout(
        title="Lift / Cumulative Gains Chart",
        xaxis_title="% of customers targeted (ranked by predicted risk)",
        yaxis={"title": "% of churners captured", "range": [0, 102]},
        yaxis2={
            "title": "Lift",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
        },
    )
    return apply_theme(fig)
