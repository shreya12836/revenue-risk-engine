"""Loads and caches the single ChurnPredictor instance the API serves.

Auto-discovers the most recently created ``outputs/*/model_v1.joblib`` by
default so the API always serves the latest trained model without a
hard-coded path; ``MODEL_ARTIFACTS_DIR`` overrides this for deployment or
testing when a specific artifacts folder must be pinned.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from models.predict import ChurnPredictor
from utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR_ENV_VAR = "MODEL_ARTIFACTS_DIR"
MODEL_VERSION = "v1"

ROOT = Path(__file__).resolve().parents[2]


def resolve_artifacts_dir() -> Path:
    """Resolve which artifacts directory to load the model from."""
    override = os.environ.get(ARTIFACTS_DIR_ENV_VAR)
    if override:
        artifacts_dir = Path(override)
        if not (artifacts_dir / f"model_{MODEL_VERSION}.joblib").exists():
            raise FileNotFoundError(
                f"{ARTIFACTS_DIR_ENV_VAR}={artifacts_dir!s} has no "
                f"model_{MODEL_VERSION}.joblib"
            )
        return artifacts_dir

    candidates = sorted(
        (ROOT / "outputs").glob(f"*/model_{MODEL_VERSION}.joblib"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No trained model artifacts found under outputs/*/model_v1.joblib. "
            f"Train a model first, or set {ARTIFACTS_DIR_ENV_VAR} explicitly."
        )
    return candidates[-1].parent


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    """Load and cache the one ChurnPredictor instance the whole API shares.

    Cached with ``lru_cache`` so every request reuses the same in-memory
    model instead of re-reading artifacts from disk; ``main.py``'s lifespan
    hook calls this once eagerly at startup so a broken/missing artifacts
    directory fails fast at boot instead of on the first request.
    """
    artifacts_dir = resolve_artifacts_dir()
    logger.info("Loading ChurnPredictor artifacts from %s", artifacts_dir)
    return ChurnPredictor.from_artifacts(artifacts_dir, version=MODEL_VERSION)
