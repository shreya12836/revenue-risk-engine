"""Page 2 -- Revenue at Risk.

No persisted predictions artifact exists, so this page live-scores the
current customer population (cached) via ``services.scoring``. All filtering
below runs client-side with pandas on the already-cached DataFrame -- moving
a slider never re-triggers scoring.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from services.artifacts import render_sidebar_and_guard
from services.scoring import score_customer_population
from utils.logger import get_logger

logger = get_logger(__name__)

artifacts_dir_str = render_sidebar_and_guard()

st.title("Revenue at Risk")

try:
    population = score_customer_population(artifacts_dir_str)
except Exception:
    logger.exception("Failed to score the current customer population")
    st.error(
        "Could not load the customer transaction dataset needed to score the "
        "current population. Confirm `data/online_retail_II.xlsx` exists or is "
        "reachable, then reload this page."
    )
    st.stop()

customers = population.customers
st.caption(
    f"Customers with purchase history as of the test snapshot "
    f"({population.snapshot_date}) -- not an all-time customer count."
)

st.metric("Total customers", f"{len(customers):,}")

st.subheader("Churn probability distribution")
hist_fig = px.histogram(
    customers,
    x="churn_probability",
    nbins=30,
    title="Distribution of predicted churn probability",
    labels={"churn_probability": "Churn probability"},
)
st.plotly_chart(hist_fig, width="stretch")

st.subheader("High-risk customers")

threshold = st.slider("Minimum churn probability", 0.0, 1.0, 0.5, step=0.01)
filtered = customers[customers["churn_probability"] >= threshold]

search_query = st.text_input("Search by customer ID")
if search_query:
    filtered = filtered[filtered["customer_id"].astype(str).str.contains(search_query, case=False)]

# Bounds computed from the full unfiltered population so this slider's own
# range doesn't shift as the threshold slider above it changes.
revenue_min = float(customers["revenue_at_risk"].min())
revenue_max = float(customers["revenue_at_risk"].max())
revenue_range = st.slider(
    "Revenue-at-risk range (£)",
    min_value=revenue_min,
    max_value=revenue_max,
    value=(revenue_min, revenue_max),
)
filtered = filtered[
    filtered["revenue_at_risk"].between(revenue_range[0], revenue_range[1])
]

default_columns = ["customer_id", "churn_probability", "spend_90d", "revenue_at_risk"]
available_columns = list(customers.columns)
selected_columns = st.multiselect(
    "Columns to display", options=available_columns, default=default_columns
)
display_columns = selected_columns or default_columns

display_df = filtered.sort_values("revenue_at_risk", ascending=False)[display_columns]
st.dataframe(
    display_df,
    width="stretch",
    column_config={
        "churn_probability": st.column_config.NumberColumn(format="%.3f"),
        "revenue_at_risk": st.column_config.NumberColumn(format="£%.2f"),
        "spend_90d": st.column_config.NumberColumn(format="£%.2f"),
    },
)

st.download_button(
    "Download filtered table as CSV",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="high_risk_customers.csv",
    mime="text/csv",
)

col1, col2 = st.columns(2)
with col1:
    st.metric("Revenue at risk (filtered)", f"£{filtered['revenue_at_risk'].sum():,.0f}")
with col2:
    st.metric("Revenue at risk (all customers)", f"£{customers['revenue_at_risk'].sum():,.0f}")

st.caption(
    "Revenue-at-risk uses trailing 90-day spend as a proxy for customer "
    "value (no trained CLV model exists yet) -- see the project README for "
    "the full methodology."
)
