"""Dynamic calibration curve, computed live from test-set predictions."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from components.theme import ACCENT, REFERENCE_LINE, apply_theme


def calibration_curve_figure(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> go.Figure:
    """Reliability diagram: mean predicted probability vs. observed churn rate per bin.

    Marker size scales with the number of customers in each bin, so a
    reader can tell a well-calibrated-but-sparse bin from a solid one.
    """
    n_bins = min(n_bins, max(2, len(np.unique(y_proba))))
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")

    bin_edges = np.quantile(y_proba, np.linspace(0, 1, n_bins + 1))
    bin_counts = np.histogram(y_proba, bins=bin_edges)[0]
    bin_counts = bin_counts[: len(prob_pred)] if len(bin_counts) >= len(prob_pred) else bin_counts

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="markers+lines",
            name="Tuned XGBoost",
            line={"color": ACCENT},
            marker={"size": np.clip(bin_counts, 6, 30) if len(bin_counts) else 8},
            customdata=bin_counts if len(bin_counts) else None,
            hovertemplate=(
                "Mean predicted: %{x:.3f}<br>Observed rate: %{y:.3f}<br>"
                "n = %{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line={"color": REFERENCE_LINE, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed churn rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return apply_theme(fig)
