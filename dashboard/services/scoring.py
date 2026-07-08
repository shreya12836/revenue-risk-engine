"""Live-scoring pipeline for the current customer population (Page 2).

No persisted predictions artifact exists (``docs/roadmap.md`` explicitly
defers a "batch scoring CLI" to post-MVP), so this pipeline reuses the same
leakage-safe feature pipeline and ``ChurnPredictor`` the training script uses,
scoring the as-of-test-snapshot customer population on demand. Broken into
small functions so each step is testable without a live model or the full
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from data.cleaner import clean
from data.loader import load
from features.builder import build_features
from models.evaluate import revenue_at_risk
from models.predict import ChurnPredictor
from services.artifacts import DEFAULT_CONFIG_PATH, get_cached_predictor
from utils.config import ProjectConfig, load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def load_customer_population(config: ProjectConfig) -> pd.DataFrame:
    """Load -> clean -> build_features(test snapshot).

    Returns ``customer_id`` + the 33 feature columns for every customer with
    purchase history as of ``config.features.snapshot_dates["test"]``. Lets
    ``FileNotFoundError``/download errors from ``data.loader.load`` propagate
    -- the page distinguishes this failure from a missing-model-artifacts
    guard and shows its own message.
    """
    raw = load(config)
    cleaned = clean(raw, config)
    snapshot = config.features.snapshot_dates["test"]
    return build_features(cleaned, config, snapshot_date=snapshot)


def score_population(predictor: ChurnPredictor, features_df: pd.DataFrame) -> np.ndarray:
    """Score every row in ``features_df`` (excluding the ``customer_id`` join key)."""
    X = features_df.drop(columns=["customer_id"])
    return predictor.predict_proba(X)


def attach_revenue_at_risk(features_df: pd.DataFrame, churn_proba: np.ndarray) -> pd.DataFrame:
    """Attach ``churn_probability`` and ``revenue_at_risk`` columns.

    Revenue-at-risk is ``churn_probability x spend_90d`` -- trailing 90-day
    spend, the same proxy ``models.train.run_training`` uses in the absence
    of a trained CLV model. ``spend_90d`` is filled with 0.0 defensively,
    mirroring ``train.py``'s own handling of this value, even though it is a
    required (non-nullable) feature column and should not contain NaN.
    """
    scored = features_df.copy()
    scored["churn_probability"] = churn_proba
    value_proxy = scored["spend_90d"].fillna(0.0)
    scored["revenue_at_risk"] = revenue_at_risk(churn_proba, value_proxy)
    return scored


@dataclass(frozen=True)
class ScoredPopulation:
    """The current customer population, scored and revenue-weighted."""

    customers: pd.DataFrame
    snapshot_date: str
    artifacts_dir: str


@st.cache_data(show_spinner="Scoring current customer population...")
def score_customer_population(
    artifacts_dir_str: str, config_path: str = DEFAULT_CONFIG_PATH
) -> ScoredPopulation:
    """Orchestrate the full live-scoring pipeline. The only cached, expensive entry point.

    Reads and processes the full transaction dataset -- expect real wall-clock
    time (tens of seconds) on first call per artifacts directory. Cached so
    repeated reruns (e.g. moving a filter slider) never re-trigger this.
    """
    config = load_config(config_path)
    predictor = get_cached_predictor(artifacts_dir_str)

    features_df = load_customer_population(config)
    churn_proba = score_population(predictor, features_df)
    scored = attach_revenue_at_risk(features_df, churn_proba)

    return ScoredPopulation(
        customers=scored,
        snapshot_date=config.features.snapshot_dates["test"],
        artifacts_dir=artifacts_dir_str,
    )
