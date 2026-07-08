"""Page 1 -- Model Performance.

Reads persisted artifacts (``metrics.json``, ``feature_importance.csv``, the
pre-rendered ROC/PR/SHAP PNGs) rather than recomputing anything -- the ROC/PR
curves and SHAP summary are exactly the plots ``models/evaluate.py`` and
``models/explain.py`` already produced during training.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from components.confusion_matrix import confusion_matrix_figure
from components.metric_cards import render_metric_row
from services.artifacts import (
    figure_path,
    load_feature_importance,
    load_metrics,
    render_sidebar_and_guard,
)

artifacts_dir_str = render_sidebar_and_guard()

st.title("Model Performance")

metrics = load_metrics(artifacts_dir_str)
tuned = metrics["xgboost_tuned"]

st.caption("Headline metrics for the tuned XGBoost model (the model `model_v1.joblib` serves).")
render_metric_row(tuned, ["roc_auc", "pr_auc", "precision", "recall", "f1"])

with st.expander("Compare against baseline / untuned XGBoost"):
    st.dataframe(pd.DataFrame(metrics).T)

st.subheader("ROC and Precision-Recall curves")
col1, col2 = st.columns(2)
with col1:
    roc_path = figure_path(artifacts_dir_str, "xgboost_tuned_roc_curve.png")
    if roc_path:
        st.image(str(roc_path), caption="ROC curve (tuned XGBoost)")
    else:
        st.info("ROC curve figure not available for this artifacts directory.")
with col2:
    pr_path = figure_path(artifacts_dir_str, "xgboost_tuned_pr_curve.png")
    if pr_path:
        st.image(str(pr_path), caption="Precision-Recall curve (tuned XGBoost)")
    else:
        st.info("PR curve figure not available for this artifacts directory.")

st.subheader("Confusion matrix")
confusion_fig = confusion_matrix_figure(tuned["confusion_matrix"])
st.pyplot(confusion_fig)
plt.close(confusion_fig)

st.subheader("Feature importance")
importance_df = load_feature_importance(artifacts_dir_str)
top15 = importance_df.sort_values("mean_abs_shap", ascending=False).head(15)
top15 = top15.sort_values("mean_abs_shap")  # ascending, so the largest bar renders on top
bar_fig = px.bar(
    top15,
    x="mean_abs_shap",
    y="feature",
    orientation="h",
    title="Top 15 features by mean |SHAP|",
    labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
)
st.plotly_chart(bar_fig, width="stretch")

shap_summary_path = figure_path(artifacts_dir_str, "shap_summary.png")
if shap_summary_path:
    st.image(
        str(shap_summary_path),
        caption="SHAP summary -- global feature impact across all customers",
    )
else:
    st.info("SHAP summary figure not available for this artifacts directory.")
