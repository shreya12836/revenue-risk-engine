"""Schema-validated inference wrapper for the tuned churn model (Day 6).

``feature_schema.json`` backs ``ChurnPredictor.predict_proba``'s boundary
validation — the DataFrame-column check Pydantic can't express for a
tabular payload. This is also the first building block toward the
FastAPI inference endpoint already on the README roadmap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


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

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        expected = set(self.schema["feature_columns"])
        actual = set(X.columns)

        missing = expected - actual
        if missing:
            raise ValueError(f"Missing required feature column(s): {sorted(missing)}")

        extra = actual - expected
        if extra:
            raise ValueError(f"Unexpected feature column(s): {sorted(extra)}")

        X_ordered = X[self.schema["feature_columns"]]
        return self.model.predict_proba(X_ordered)[:, 1]
