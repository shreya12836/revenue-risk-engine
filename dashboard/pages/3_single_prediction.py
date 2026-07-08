"""Page 3 -- Single Customer Prediction.

Builds the form from ``api.schemas``'s feature-column constants (rather than
re-deriving the feature list), validates through the real ``PredictionRequest``
model, and scores via the same ``ChurnPredictor`` the FastAPI service uses.
"""

from __future__ import annotations

from pydantic import ValidationError

import streamlit as st

from api.schemas import NULLABLE_FEATURE_COLUMNS, REQUIRED_FEATURE_COLUMNS
from components.shap_chart import render_waterfall
from services.artifacts import get_cached_predictor, render_sidebar_and_guard
from services.single_customer import (
    build_prediction_request,
    explain_single_customer,
    score_single_customer,
)

# Nullable fields with no ge=0 constraint in PredictionRequest -- trend
# slopes can legitimately be negative (a declining trend).
UNBOUNDED_NULLABLE_FIELDS = {"spend_slope", "txn_count_slope"}

artifacts_dir_str = render_sidebar_and_guard()

st.title("Single Customer Prediction")
st.caption(
    "Enter a customer's feature vector to score their churn probability and "
    "see which features drove the prediction."
)

with st.form("single_customer_form"):
    customer_id = st.text_input(
        "Customer ID (optional)",
        help="Echoed back only -- not used for any lookup.",
    )

    st.subheader("Required features")
    required_values: dict[str, float] = {}
    for field in REQUIRED_FEATURE_COLUMNS:
        required_values[field] = st.number_input(field, min_value=0.0, value=0.0, step=1.0)

    nullable_values: dict[str, float | None] = {}
    with st.expander("Optional / statistical features (leave unchecked if not applicable)"):
        for field in NULLABLE_FEATURE_COLUMNS:
            available = st.checkbox(f"{field} available?", value=False, key=f"has_{field}")
            if available:
                min_value = None if field in UNBOUNDED_NULLABLE_FIELDS else 0.0
                nullable_values[field] = st.number_input(
                    field, min_value=min_value, value=0.0, step=1.0, key=f"val_{field}"
                )
            else:
                nullable_values[field] = None

    submitted = st.form_submit_button("Predict")

if not submitted:
    st.stop()

form_values = {**required_values, **nullable_values}

try:
    request = build_prediction_request(form_values, customer_id or None)
except ValidationError as exc:
    for error in exc.errors():
        field_name = error["loc"][0] if error["loc"] else "input"
        st.error(f"{field_name}: {error['msg']}")
    st.stop()

predictor = get_cached_predictor(artifacts_dir_str)

try:
    churn_probability = score_single_customer(predictor, request)
except ValueError as exc:
    st.error(f"Could not score this customer: {exc}")
    st.stop()

st.subheader("Result")
st.metric("Churn probability", f"{churn_probability:.1%}")

if churn_probability >= 0.66:
    risk_band = "high"
elif churn_probability >= 0.33:
    risk_band = "medium"
else:
    risk_band = "low"
st.write(f"This customer is at **{risk_band} risk** of churning in the next period.")

st.info(
    "Customer lifetime value model not yet available -- deferred per the "
    "project roadmap. Revenue-at-risk elsewhere in this dashboard uses "
    "90-day spend as a proxy for customer value."
)

st.subheader("Why this prediction")
shap_values = explain_single_customer(predictor, request)
if shap_values is not None:
    render_waterfall(shap_values[0])
else:
    st.warning(
        "Prediction succeeded, but the SHAP explanation couldn't be generated "
        "for this input. Showing the prediction result above without a "
        "feature-level breakdown."
    )
