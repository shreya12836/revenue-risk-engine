"""Assemble model-ready feature/label matrices from a ``FeatureLabelPair``.

Kept separate from ``features.splits`` because it's a modeling-layer
concern (which target, which columns become ``X``) rather than a
feature-engineering one.
"""

from __future__ import annotations

import pandas as pd

from features.splits import FeatureLabelPair
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_TARGETS = ("churn", "clv")


def prepare_xy(
    pair: FeatureLabelPair, target: str = "churn"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` ready for model fitting.

    Joins features to labels on ``customer_id``, drops customers with no
    ``target`` label (they entered the population too close to the snapshot
    to have an observed outcome — see ``FeatureLabelPair.joined``), and
    separates the join key and label columns from the feature matrix.
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

    joined = pair.joined
    labeled = joined.dropna(subset=[target])

    dropped = len(joined) - len(labeled)
    if dropped:
        logger.info(
            "prepare_xy: dropped %d customers with no %s label", dropped, target
        )

    feature_columns = [c for c in pair.features.columns if c != "customer_id"]
    X = labeled[feature_columns].reset_index(drop=True)
    y = labeled[target].reset_index(drop=True)
    if target == "churn":
        y = y.astype(int)
    return X, y
