"""Page 1 -- Model Performance.

Every chart here is generated at runtime -- either from the live-scored
held-out test set (``services.evaluation.evaluate_test_set``, since no raw
prediction array is ever persisted to ``outputs/<ts>/``) or from artifact
aggregates (``metrics.json``, ``feature_importance.csv``) that are cheap and
already correct to read as-is. No PNGs are embedded anywhere on this page.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from components.calibration_chart import calibration_curve_figure
from components.confusion_matrix import confusion_matrix_figure
from components.lift_chart import lift_gains_figure
from components.metric_cards import render_metric_row
from components.model_comparison import model_comparison_figure
from components.roc_pr_curves import pr_curve_figure, roc_curve_figure
from services.artifacts import load_feature_importance, load_metrics, render_sidebar_and_guard
from services.evaluation import EvaluatedTestSet, evaluate_test_set
from utils.logger import get_logger

logger = get_logger(__name__)

artifacts_dir_str = render_sidebar_and_guard()

st.title("Model Performance")
st.caption(
    "All metrics and charts below are computed live from the active artifacts "
    "directory and a fresh scoring pass over the held-out test split -- "
    "nothing on this page is a static image."
)

try:
    metrics = load_metrics(artifacts_dir_str)
except json.JSONDecodeError:
    st.error(
        "`metrics.json` could not be parsed -- this artifacts directory may be "
        "corrupt. Re-run training, or point `MODEL_ARTIFACTS_DIR` elsewhere."
    )
    st.stop()

tuned = metrics.get("xgboost_tuned")
if tuned is None:
    st.error(
        "`metrics.json` is missing the 'xgboost_tuned' entry -- this artifacts "
        "directory may be from an incompatible training run."
    )
else:
    st.caption(
        "Headline metrics for the tuned XGBoost model (the model `model_v1.joblib` serves). "
        "Computed on a single held-out test split -- no cross-validation."
    )
    render_metric_row(tuned, ["roc_auc", "pr_auc", "precision", "recall", "f1"])

st.subheader("Model comparison")
if len(metrics) >= 2:
    st.plotly_chart(model_comparison_figure(metrics), width="stretch")

    default = metrics.get("xgboost_default")
    if tuned is not None and default is not None:
        precision_delta = tuned["precision"] - default["precision"]
        recall_delta = tuned["recall"] - default["recall"]
        direction = "gains" if recall_delta >= 0 else "loses"
        cost = "gives up" if precision_delta < 0 else "keeps"
        st.markdown(
            f"**Precision vs. recall tradeoff:** tuning {direction} "
            f"{abs(recall_delta):.1%} recall (catches that many more churners) "
            f"while it {cost} {abs(precision_delta):.1%} precision "
            f"(that's the change in false-alarm rate among flagged customers) "
            f"versus the untuned default model. In practice: a higher-recall "
            f"model finds more true churners but sends more false alerts to "
            f"the retention team; the right tradeoff depends on the cost of "
            f"contacting a customer who wasn't actually going to churn."
        )
else:
    st.info("Only one model's metrics are available in this artifacts directory.")

evaluated: EvaluatedTestSet | None
try:
    evaluated = evaluate_test_set(artifacts_dir_str)
except Exception:
    logger.exception("Failed to reconstruct/score the held-out test set")
    evaluated = None
    st.error(
        "Could not reconstruct the held-out test set needed for the ROC, "
        "Precision-Recall, calibration, lift, and churn-probability charts. "
        "Confirm `data/online_retail_II.xlsx` exists or is reachable, then "
        "reload this page. (Confusion matrix, feature importance, and model "
        "comparison above are unaffected -- they come from persisted artifacts.)"
    )

if evaluated is not None:
    st.subheader("ROC and Precision-Recall curves")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(roc_curve_figure(evaluated.y_true, evaluated.y_proba), width="stretch")
    with col2:
        st.plotly_chart(pr_curve_figure(evaluated.y_true, evaluated.y_proba), width="stretch")

if tuned is not None:
    st.subheader("Confusion matrix")
    st.plotly_chart(confusion_matrix_figure(tuned["confusion_matrix"]), width="stretch")
    st.caption(
        "Rows are actual outcomes, columns are predicted outcomes at the "
        "default 0.5 probability threshold used when `metrics.json` was written."
    )

if evaluated is not None:
    st.subheader("Calibration curve")
    st.plotly_chart(
        calibration_curve_figure(evaluated.y_true, evaluated.y_proba), width="stretch"
    )
    st.caption(
        "A well-calibrated model's points sit on the diagonal: when it says "
        "'70% likely to churn,' about 70% of those customers actually do."
    )

    st.subheader("Lift / cumulative gains")
    st.plotly_chart(lift_gains_figure(evaluated.y_true, evaluated.y_proba), width="stretch")
    st.caption(
        "Shows how much better than random targeting the model is: e.g. "
        "contacting the top 20% highest-risk customers catches far more than "
        "20% of actual churners."
    )

    st.subheader("Churn probability distribution (test set)")
    threshold = st.session_state.get("risk_threshold", 0.5)
    hist_fig = px.histogram(
        x=evaluated.y_proba,
        nbins=30,
        title="Distribution of predicted churn probability -- held-out test set",
        labels={"x": "Churn probability"},
    )
    hist_fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#C0392B",
        annotation_text=f"Revenue-at-risk threshold ({threshold:.2f})",
    )
    st.plotly_chart(hist_fig, width="stretch")
    st.caption(
        "The dashed line marks the same risk threshold currently set on the "
        "Revenue at Risk page, so the two views stay comparable."
    )

st.subheader("Feature importance")
try:
    importance_df = load_feature_importance(artifacts_dir_str)
except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
    importance_df = pd.DataFrame()

if importance_df.empty:
    st.warning("Feature importance data not available for this artifacts directory.")
else:
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
    st.caption(
        "Higher bars mean the feature has more influence, on average, over "
        "the model's churn predictions across all customers (global "
        "importance) -- see the Single Customer Prediction page for how a "
        "feature affects one specific prediction."
    )
