"""SHAP explainability for the tuned XGBoost model (Day 6).

Uses ``shap.TreeExplainer`` — an exact, fast explainer for tree ensembles,
rather than a model-agnostic approximation. Only the tuned model is
explained (not the baseline or default XGBoost): the tuned model is the
one that ships, so its explanations are the ones that matter for the
business narrative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for every row in ``X`` via ``TreeExplainer``."""
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def save_shap_summary_plot(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a SHAP beeswarm summary plot (global feature impact)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def save_shap_waterfall_plot(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    customer_ids: pd.Series,
    customer_id: int,
    output_path: str | Path,
) -> Path:
    """Save a SHAP waterfall plot explaining one customer's prediction.

    ``customer_ids`` must be row-aligned with ``X``/``shap_values`` (the
    caller's ``X`` typically excludes the id column, so it's passed
    separately here rather than looked up from ``X`` itself).
    """
    matches = np.flatnonzero(customer_ids.to_numpy() == customer_id)
    if matches.size == 0:
        raise ValueError(f"customer_id {customer_id!r} not found in customer_ids")
    idx = int(matches[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap.plots.waterfall(shap_values[idx], show=False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def build_feature_importance_table(
    model: XGBClassifier,
    shap_values: shap.Explanation,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by both XGBoost-native and SHAP-based importance.

    Two perspectives are kept side by side deliberately: XGBoost's native
    importance is model-internal (split gain), while mean absolute SHAP is
    model-agnostic and reflects actual prediction impact on this data.
    """
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "xgboost_importance": model.feature_importances_,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        }
    )
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
