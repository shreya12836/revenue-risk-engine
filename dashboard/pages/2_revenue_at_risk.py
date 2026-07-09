"""Page 2 -- Revenue at Risk.

No persisted predictions artifact exists, so this page live-scores the
current customer population (cached) via ``services.scoring``. All filtering
below runs client-side with pandas on the already-cached DataFrame -- moving
a slider never re-triggers scoring.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.band_chart import band_risk_figure
from components.cumulative_risk_chart import cumulative_revenue_at_risk_figure
from components.driver_bar_chart import driver_bar_figure
from components.pareto_chart import pareto_figure
from formatting import format_count, format_currency
from services.artifacts import render_sidebar_and_guard
from services.explainability import explain_population, humanize_feature_name, top_churn_drivers
from services.risk_bands import band_risk_summary
from services.scoring import score_customer_population
from utils.logger import get_logger

logger = get_logger(__name__)

BAND_COLUMNS = {
    "Spend (90d)": "spend_90d",
    "Recency": "recency_days",
    "Frequency": "frequency",
    "Tenure": "tenure_days",
}
SEGMENT_COLUMN = "segment"

DEFAULT_THRESHOLD = 0.5
DEFAULT_SEARCH = ""
DEFAULT_TOP_N = 20
DEFAULT_COLUMNS = ["customer_id", "churn_probability", "spend_90d", "revenue_at_risk"]

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

revenue_min = float(customers["revenue_at_risk"].min())
revenue_max = float(customers["revenue_at_risk"].max())

if st.button("Reset filters", help="Clear threshold, search, revenue range, and column selections"):
    st.session_state["risk_threshold"] = DEFAULT_THRESHOLD
    st.session_state["customer_search"] = DEFAULT_SEARCH
    st.session_state["revenue_range"] = (revenue_min, revenue_max)
    st.session_state["display_columns"] = DEFAULT_COLUMNS
    st.rerun()

st.subheader("Churn probability distribution")
hist_fig = px.histogram(
    customers,
    x="churn_probability",
    nbins=30,
    title="Distribution of predicted churn probability",
    labels={"churn_probability": "Churn probability"},
)
st.plotly_chart(hist_fig, width="stretch")

threshold = st.slider(
    "Minimum churn probability (risk threshold)",
    0.0,
    1.0,
    DEFAULT_THRESHOLD,
    step=0.01,
    key="risk_threshold",
    help="Customers at or above this predicted churn probability are treated as 'high-risk' below.",
)
high_risk = customers[customers["churn_probability"] >= threshold]

st.subheader("Key metrics")
kpi_cols = st.columns(4)
kpi_cols[0].metric(
    "Total customers", format_count(len(customers)), help="All customers in the current population."
)
kpi_cols[1].metric(
    "High-risk customers",
    format_count(len(high_risk)),
    help=f"Customers with churn probability >= {threshold:.2f}.",
)
kpi_cols[2].metric(
    "Total revenue at risk",
    format_currency(customers["revenue_at_risk"].sum()),
    help="Sum of churn_probability x 90-day spend across all customers.",
)
avg_high_risk_revenue = high_risk["revenue_at_risk"].mean() if len(high_risk) else None
kpi_cols[3].metric(
    "Avg revenue at risk (high-risk)",
    format_currency(avg_high_risk_revenue),
    help="Average revenue at risk per customer, among high-risk customers only.",
)

st.subheader("Revenue-at-risk concentration")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(cumulative_revenue_at_risk_figure(customers), width="stretch")
with col2:
    st.plotly_chart(pareto_figure(customers), width="stretch")
st.caption(
    "These show how concentrated revenue risk is: if a small share of "
    "customers accounts for most of the revenue at risk, retention effort "
    "targeted at them goes further."
)

st.subheader("Risk by customer segment")
for label, column in BAND_COLUMNS.items():
    if column not in customers.columns:
        continue
    summary = band_risk_summary(customers, column)
    st.plotly_chart(band_risk_figure(summary, f"Risk by {label} band"), width="stretch")

if SEGMENT_COLUMN in customers.columns:
    segment_summary = band_risk_summary(customers, SEGMENT_COLUMN)
    st.plotly_chart(band_risk_figure(segment_summary, "Risk by customer segment"), width="stretch")

with st.expander("Top churn drivers for high-risk customers", expanded=False):
    st.caption(
        "Computes SHAP values for the entire current population (may take a "
        "few seconds the first time)."
    )
    try:
        pop_explanation = explain_population(artifacts_dir_str)
    except Exception:
        logger.exception("Failed to compute population-level SHAP explanations")
        st.warning("Could not compute churn drivers for the current population.")
    else:
        mask = pop_explanation.customers["churn_probability"] >= threshold
        drivers = top_churn_drivers(pop_explanation, mask)
        if drivers.empty:
            st.info("No customers match the current risk threshold.")
        else:
            st.plotly_chart(driver_bar_figure(drivers), width="stretch")
            top_labels = [humanize_feature_name(f) for f in drivers["feature"].head(3)]
            st.markdown(
                f"Across current high-risk customers, **{', '.join(top_labels)}** "
                "are the strongest churn signals."
            )

st.subheader("High-risk customers")

search_query = st.text_input("Search by customer ID", key="customer_search")
filtered = high_risk
if search_query:
    filtered = filtered[filtered["customer_id"].astype(str).str.contains(search_query, case=False)]

# Bounds computed from the full unfiltered population so this slider's own
# range doesn't shift as the threshold slider above it changes.
revenue_range = st.slider(
    "Revenue-at-risk range (£)",
    min_value=revenue_min,
    max_value=revenue_max,
    value=(revenue_min, revenue_max),
    key="revenue_range",
)
filtered = filtered[filtered["revenue_at_risk"].between(revenue_range[0], revenue_range[1])]

available_columns = list(customers.columns)
selected_columns = st.multiselect(
    "Columns to display",
    options=available_columns,
    default=DEFAULT_COLUMNS,
    key="display_columns",
)
display_columns = selected_columns or DEFAULT_COLUMNS

if filtered.empty:
    st.info("No customers match the current filters.")
else:
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

    st.subheader("Top N high-risk customers")
    top_n = st.number_input(
        "Show top N by revenue at risk", min_value=5, max_value=200, value=DEFAULT_TOP_N, step=5
    )
    top_n_df = filtered.sort_values("revenue_at_risk", ascending=False).head(int(top_n))[
        display_columns
    ]
    st.dataframe(top_n_df, width="stretch")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Revenue at risk (filtered)",
        format_currency(filtered["revenue_at_risk"].sum()) if not filtered.empty else "£0",
    )
with col2:
    st.metric(
        "Revenue at risk (all customers)",
        format_currency(customers["revenue_at_risk"].sum()),
    )

st.caption(
    "Revenue-at-risk uses trailing 90-day spend as a proxy for customer "
    "value (no trained CLV model exists yet) -- see the project README for "
    "the full methodology."
)
