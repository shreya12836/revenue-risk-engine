"""Model versioning: reproducibility metadata and versioned artifacts.

``metadata.json`` exists so a deployed model can always be traced back to
the exact commit, hyperparameters, and data sizes it was trained from —
"this model was trained at commit abc123" as a verifiable fact, not a
comment.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import optuna
from xgboost import XGBClassifier


def get_git_commit_hash() -> str:
    """Return the current commit hash, or ``"unknown"`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_metadata(
    model: XGBClassifier,
    feature_count: int,
    split_sizes: dict[str, int],
    study: optuna.Study,
    cv_folds: int,
    version: str = "v1",
) -> dict:
    """Assemble the reproducibility metadata for a versioned model.

    ``split_sizes`` must have ``"train"``, ``"val"``, ``"test"`` keys.
    """
    return {
        "version": version,
        "git_commit": get_git_commit_hash(),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "algorithm": "XGBoost",
        "feature_count": feature_count,
        "train_samples": split_sizes["train"],
        "val_samples": split_sizes["val"],
        "test_samples": split_sizes["test"],
        "cv_folds": cv_folds,
        "optuna_trials": len(study.trials),
        "optuna_best_pr_auc": study.best_value,
        "hyperparameters": model.get_params(),
    }


def save_model_version(
    model: XGBClassifier,
    metadata: dict,
    output_dir: str | Path,
    version: str = "v1",
) -> Path:
    """Persist the model and its metadata into ``output_dir``.

    Returns the path to the saved model file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"model_{version}.joblib"
    joblib.dump(model, model_path)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    return model_path
