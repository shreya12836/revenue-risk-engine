"""Portfolio-level and business-readable SHAP explainability.

Goes beyond a single customer's waterfall: aggregates SHAP contributions
across the current (live-scored) customer population to find the strongest
churn drivers among high-risk customers, and turns any single-row SHAP
explanation into plain-language sentences instead of raw feature names and
signed floats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import streamlit as st

from services.artifacts import get_cached_predictor
from services.scoring import score_customer_population
from utils.logger import get_logger

logger = get_logger(__name__)


def humanize_feature_name(name: str) -> str:
    """``"spend_90d"`` -> ``"Spend 90d"``; a light default label, not a lookup table."""
    words = re.sub(r"(\d+)", r" \1", name.replace("_", " ")).split()
    return " ".join(w.capitalize() for w in words)


@dataclass(frozen=True)
class PopulationExplanation:
    """Batch SHAP values for the currently live-scored customer population."""

    shap_values: shap.Explanation
    customers: pd.DataFrame
    artifacts_dir: str


@st.cache_data(show_spinner="Computing SHAP values for the current population...")
def explain_population(artifacts_dir_str: str) -> PopulationExplanation:
    """Batch-explain every customer in the current population.

    A second, separately cached (and heavier) call from
    ``score_customer_population`` -- population-wide SHAP is more expensive
    than scoring alone, so it's only triggered on demand (e.g. an expander),
    not eagerly on every page load.
    """
    population = score_customer_population(artifacts_dir_str)
    predictor = get_cached_predictor(artifacts_dir_str)
    X = population.customers.drop(
        columns=["churn_probability", "revenue_at_risk", "customer_id"]
    )
    _, shap_values = predictor.predict_with_shap(X)
    return PopulationExplanation(
        shap_values=shap_values,
        customers=population.customers,
        artifacts_dir=artifacts_dir_str,
    )


def top_churn_drivers(
    pop_explanation: PopulationExplanation, mask: pd.Series, top_n: int = 10
) -> pd.DataFrame:
    """Mean |SHAP| per feature, restricted to rows where ``mask`` is ``True``.

    ``mask`` must be row-aligned with ``pop_explanation.customers`` (e.g.
    ``customers["churn_probability"] >= threshold``). Returns an empty
    DataFrame if no rows match, rather than raising.
    """
    mask_array = np.asarray(mask)
    if not mask_array.any():
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    values = np.asarray(pop_explanation.shap_values.values)[mask_array]
    feature_names = list(pop_explanation.shap_values.feature_names)
    mean_abs = np.abs(values).mean(axis=0)

    drivers = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return drivers.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def business_readable_explanation(
    shap_row: shap.Explanation,
    feature_labels: dict[str, str] | None = None,
    top_k: int = 5,
) -> list[str]:
    """Turn a single-row SHAP explanation into plain-language sentences.

    Every sentence references only values present in ``shap_row`` (raw
    feature value + sign/rank of its SHAP contribution) -- no fabricated
    numbers. ``feature_labels`` overrides ``humanize_feature_name`` for
    specific columns; unlisted columns fall back to the generated label.
    """
    feature_labels = feature_labels or {}
    values = np.asarray(shap_row.values)
    data = np.asarray(shap_row.data)
    feature_names = list(shap_row.feature_names)

    order = np.argsort(-np.abs(values))[:top_k]
    sentences: list[str] = []
    for idx in order:
        name = feature_names[idx]
        label = feature_labels.get(name, humanize_feature_name(name))
        direction = "raising" if values[idx] > 0 else "lowering"
        sentences.append(f"{label} ({data[idx]:g}) is {direction} this customer's churn risk.")
    return sentences
