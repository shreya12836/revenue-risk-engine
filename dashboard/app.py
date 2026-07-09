"""Revenue Risk Engine dashboard -- entrypoint.

Run with ``make run-dashboard`` or ``streamlit run dashboard/app.py``. Every
page under ``pages/`` scores through the same ``ChurnPredictor`` the training
pipeline and FastAPI service use (``src/models/predict.py``) -- this
dashboard never retrains a model or duplicates inference logic, it only reads
persisted artifacts and calls the shared predictor.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from services.artifacts import render_sidebar_and_guard
except ImportError:
    # Defensive fallback only: the project is normally installed editable
    # (`pip install -e ".[dev]"`, per the Makefile's `install` target), which
    # already puts `src/` on the import path the same way `src/api/main.py`
    # relies on. This only matters if the dashboard is ever launched without
    # that install having run.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from services.artifacts import render_sidebar_and_guard  # noqa: E402

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Revenue Risk Engine",
    page_icon="\U0001f4c9",
    layout="wide",
)

render_sidebar_and_guard()

st.title("Revenue Risk Engine")
st.markdown(
    """
A customer churn and revenue-at-risk dashboard for an e-commerce retailer,
built on top of a tuned XGBoost churn model served through a schema-validated
FastAPI inference service. Every number on these pages comes from persisted
training artifacts or a live call to the same `ChurnPredictor` the API uses
-- nothing here retrains a model.

**Use the sidebar to navigate:**

- **Model Performance** -- ROC/PR curves, confusion matrix, feature importance.
- **Revenue at Risk** -- the current customer population scored for churn risk,
  with an interactive high-risk customer table.
- **Single Customer Prediction** -- score one customer's feature vector and see
  a SHAP explanation for the prediction.
"""
)

st.caption(
    "See the sidebar's Artifact Information panel for the active model "
    "version, training date, and snapshot date."
)
