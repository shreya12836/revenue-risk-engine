"""Shared schema-validated inference wrapper for the tuned churn model.

``feature_schema.json`` backs ``ChurnPredictor``'s boundary validation —
the DataFrame-column check Pydantic can't express for a tabular payload.
This is the single inference entry point the training pipeline, the SHAP
explainability step, and the future FastAPI endpoint all share, so scoring
logic (column validation + ordering) is never duplicated between them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from models.explain import compute_shap_values


def build_feature_schema(X: pd.DataFrame, version: str = "v1") -> dict:
    """Build a schema describing the feature columns a model expects."""
    return {
        "version": version,
        "feature_columns": list(X.columns),
        "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
    }


def save_feature_schema(schema: dict, output_path: str | Path) -> Path:
    """Persist a feature schema to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return output_path


@dataclass(frozen=True)
class ChurnPredictor:
    """Loads a versioned model + its feature schema, and scores new data.

    ``predict_proba`` is the only way to score with this object: it always
    validates incoming columns against the schema and reindexes to the
    schema's column order before calling the model, so a caller can never
    accidentally score with columns in the wrong order or a silently
    missing feature.
    """

    model: XGBClassifier
    schema: dict

    @classmethod
    def from_artifacts(cls, model_dir: str | Path, version: str = "v1") -> ChurnPredictor:
        model_dir = Path(model_dir)
        model = joblib.load(model_dir / f"model_{version}.joblib")
        schema = json.loads(
            (model_dir / "feature_schema.json").read_text(encoding="utf-8")
        )
        return cls(model=model, schema=schema)

    def _validate_and_order(self, X: pd.DataFrame) -> pd.DataFrame:
        expected = set(self.schema["feature_columns"])
        actual = set(X.columns)

        missing = expected - actual
        if missing:
            raise ValueError(f"Missing required feature column(s): {sorted(missing)}")

        extra = actual - expected
        if extra:
            raise ValueError(f"Unexpected feature column(s): {sorted(extra)}")

        return X[self.schema["feature_columns"]]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_ordered = self._validate_and_order(X)
        return self.model.predict_proba(X_ordered)[:, 1]

    def predict_with_shap(self, X: pd.DataFrame) -> tuple[np.ndarray, shap.Explanation]:
        """Score ``X`` and explain every prediction with SHAP in one call.

        Runs the same schema validation/reordering as ``predict_proba`` so
        the two methods can never silently disagree on which columns (or
        column order) a given row was scored against.
        """
        X_ordered = self._validate_and_order(X)
        proba = self.model.predict_proba(X_ordered)[:, 1]
        shap_values = compute_shap_values(self.model, X_ordered)
        return proba, shap_values
