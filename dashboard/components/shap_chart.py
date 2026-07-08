"""SHAP waterfall rendering for a single customer's prediction.

Matches the exact idiom ``models/explain.py::save_shap_waterfall_plot`` uses
(``shap.plots.waterfall`` + captured pyplot figure) so the dashboard's
per-customer explanation looks like the artifact PNGs already produced by
the training pipeline, rather than inventing a second visual style.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import shap
import streamlit as st


def render_waterfall(shap_row: shap.Explanation) -> None:
    """Render a single-row SHAP explanation as a waterfall plot.

    ``shap_row`` must be a single-row ``shap.Explanation`` (e.g. ``shap_values[0]``
    from a batch of one). Closes the matplotlib figure after rendering so a
    long-lived Streamlit process never accumulates open figures across reruns.
    """
    shap.plots.waterfall(shap_row, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)
