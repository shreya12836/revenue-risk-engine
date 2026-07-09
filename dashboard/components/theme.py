"""Shared Plotly styling applied to every chart in the dashboard.

One small color/template system so risk-coded charts (ROC/PR/calibration on
Page 1, revenue-at-risk breakdowns on Page 2, SHAP on Page 3) read as one
visual system instead of each figure builder picking its own defaults.
"""

from __future__ import annotations

import plotly.graph_objects as go

RISK_LOW = "#2E8B57"
RISK_MEDIUM = "#DAA520"
RISK_HIGH = "#C0392B"
ACCENT = "#2563EB"
NEUTRAL = "#94A3B8"
REFERENCE_LINE = "#9CA3AF"

LAYOUT_DEFAULTS: dict = {
    "template": "plotly_white",
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    "hovermode": "closest",
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    "font": {"size": 13},
}


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared layout defaults to ``fig`` in place and return it."""
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig
