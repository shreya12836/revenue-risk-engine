"""Live evaluation of the held-out test split (Page 1).

No raw ``y_true``/``y_proba`` array is ever persisted to ``outputs/<ts>/`` --
``models.evaluate.save_diagnostic_plots`` only writes PNGs, and
``metrics.json`` only holds aggregates. The test split itself is fully
reproducible, though: it's built by the exact same recipe
``scripts/train.py`` uses (``load -> clean -> build_time_splits ->
prepare_xy(split.test, "churn")``), so this module reconstructs it and scores
it through the same ``ChurnPredictor`` used everywhere else in the dashboard,
giving real prediction arrays for dynamic ROC/PR/calibration/lift/histogram
charts instead of pre-baked images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from data.cleaner import clean
from data.loader import load
from features.splits import build_time_splits
from models.dataset import prepare_xy
from models.predict import ChurnPredictor
from services.artifacts import DEFAULT_CONFIG_PATH, get_cached_predictor
from utils.config import ProjectConfig, load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def load_test_split(config: ProjectConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Reconstruct ``(X_test, y_test, customer_ids_test)`` deterministically.

    Mirrors ``scripts/train.py``'s exact recipe so the dashboard's held-out
    test set is provably the same one training evaluated against, not a
    re-derivation. Lets ``FileNotFoundError``/download errors from
    ``data.loader.load`` propagate -- the page distinguishes this failure
    from a missing-model-artifacts guard, the same pattern
    ``services/scoring.py`` already uses for Page 2.
    """
    raw = load(config)
    cleaned = clean(raw, config)
    split = build_time_splits(cleaned, config)
    X_test, y_test = prepare_xy(split.test, target="churn")
    customer_ids_test = (
        split.test.joined.dropna(subset=["churn"])["customer_id"].reset_index(drop=True)
    )
    return X_test, y_test, customer_ids_test


def score_test_set(predictor: ChurnPredictor, X_test: pd.DataFrame) -> np.ndarray:
    """Score the test features through the active predictor."""
    return predictor.predict_proba(X_test)


@dataclass(frozen=True)
class EvaluatedTestSet:
    """The held-out test split, scored, for dynamic performance charts."""

    y_true: np.ndarray
    y_proba: np.ndarray
    customer_ids: np.ndarray
    artifacts_dir: str


@st.cache_data(show_spinner="Scoring held-out test set...")
def evaluate_test_set(
    artifacts_dir_str: str, config_path: str = DEFAULT_CONFIG_PATH
) -> EvaluatedTestSet:
    """Orchestrate test-split reconstruction + scoring. The only cached entry point.

    Reads and processes the full transaction dataset -- expect real
    wall-clock time on first call per artifacts directory, same accepted
    cost as ``services.scoring.score_customer_population``.
    """
    config = load_config(config_path)
    predictor = get_cached_predictor(artifacts_dir_str)

    X_test, y_test, customer_ids_test = load_test_split(config)
    y_proba = score_test_set(predictor, X_test)

    return EvaluatedTestSet(
        y_true=y_test.to_numpy(),
        y_proba=y_proba,
        customer_ids=customer_ids_test.to_numpy(),
        artifacts_dir=artifacts_dir_str,
    )
