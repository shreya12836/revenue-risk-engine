"""Artifact discovery, cached loaders, and the shared sidebar/guard.

Reuses ``api.dependencies.resolve_artifacts_dir`` so the dashboard always
serves the same "latest model" the FastAPI service would, including honoring
the ``MODEL_ARTIFACTS_DIR`` env override. Every page calls
``render_sidebar_and_guard`` as its first statement (Streamlit's classic
``pages/`` multipage convention re-executes each page script independently on
navigation, so this cannot be rendered once in ``app.py`` and expected to
persist -- it must run per page).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from api.dependencies import resolve_artifacts_dir
from models.predict import ChurnPredictor
from utils.config import ProjectConfig, load_config
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = "configs/online_retail_ii.yaml"


def resolve_active_artifacts_dir() -> Path:
    """Thin re-export of ``resolve_artifacts_dir`` for dashboard call sites."""
    return resolve_artifacts_dir()


def get_active_artifacts_dir_or_stop() -> str:
    """Resolve the active artifacts dir, or stop the page with a friendly error.

    Never returns in the failure case -- ``st.stop()`` halts script execution,
    so callers can treat the return value as always-valid.
    """
    try:
        artifacts_dir = resolve_active_artifacts_dir()
    except FileNotFoundError:
        st.error(
            "No trained model artifacts found under `outputs/`. "
            "Run `python scripts/train.py --config configs/online_retail_ii.yaml` "
            "first, or set `MODEL_ARTIFACTS_DIR` to point at a valid artifacts directory."
        )
        st.stop()
        raise  # pragma: no cover -- unreachable, st.stop() halts the script
    return str(artifacts_dir)


@st.cache_data(show_spinner=False)
def load_metadata(artifacts_dir_str: str) -> dict:
    """Read ``metadata.json`` from the given artifacts directory."""
    path = Path(artifacts_dir_str) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_metrics(artifacts_dir_str: str) -> dict[str, dict]:
    """Read ``metrics.json`` (keys: baseline, xgboost_default, xgboost_tuned)."""
    path = Path(artifacts_dir_str) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feature_importance(artifacts_dir_str: str) -> pd.DataFrame:
    """Read ``feature_importance.csv`` (feature, xgboost_importance, mean_abs_shap)."""
    path = Path(artifacts_dir_str) / "feature_importance.csv"
    return pd.read_csv(path)


def figure_path(artifacts_dir_str: str, filename: str) -> Path | None:
    """Return ``artifacts_dir/figures/filename`` if it exists, else ``None``."""
    path = Path(artifacts_dir_str) / "figures" / filename
    return path if path.exists() else None


@st.cache_resource(show_spinner=False)
def get_cached_predictor(artifacts_dir_str: str) -> ChurnPredictor:
    """Load and cache the ``ChurnPredictor`` for the given artifacts directory.

    Keyed on ``artifacts_dir_str`` (not a bare ``lru_cache``) so a new training
    run producing a newer ``outputs/<timestamp>/`` naturally busts this cache
    on the next rerun.
    """
    return ChurnPredictor.from_artifacts(artifacts_dir_str)


def build_artifact_summary(artifacts_dir_str: str, config: ProjectConfig) -> dict[str, str]:
    """Assemble the sidebar's Artifact Information fields as plain strings."""
    metadata = load_metadata(artifacts_dir_str)
    return {
        "Model version": str(metadata.get("version", "unknown")),
        "Training date": str(metadata.get("training_date", "unknown")),
        "Git commit": str(metadata.get("git_commit", "unknown")),
        "Snapshot date (customer population)": config.features.snapshot_dates["test"],
        "Active artifact directory": artifacts_dir_str,
    }


def render_artifact_sidebar(artifacts_dir_str: str) -> None:
    """Render the Artifact Information block in the sidebar.

    Collapsed by default (UX polish, not a hidden guard -- the foundational
    missing-artifacts check already happened before this is called). The raw
    filesystem path is kept out of the main block and shown only inside a
    separate "Debug info" expander, also collapsed by default.
    """
    config = load_config(DEFAULT_CONFIG_PATH)
    summary = build_artifact_summary(artifacts_dir_str, config)
    debug_only_fields = {"Active artifact directory"}

    with st.sidebar:
        with st.expander("Artifact Information", expanded=False):
            for label, value in summary.items():
                if label in debug_only_fields:
                    continue
                st.caption(f"**{label}**")
                st.text(value)
            if st.button("Refresh artifacts", help="Clear caches and re-scan outputs/"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        with st.expander("Debug info", expanded=False):
            for label in debug_only_fields:
                st.caption(f"**{label}**")
                st.code(summary[label])


def render_sidebar_and_guard() -> str:
    """Resolve the active artifacts dir, render the sidebar, return the dir string."""
    artifacts_dir_str = get_active_artifacts_dir_or_stop()
    render_artifact_sidebar(artifacts_dir_str)
    return artifacts_dir_str
