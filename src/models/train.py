"""Baseline logistic regression and XGBoost training for churn classification.

Per the roadmap, MVP trains a simple baseline before the boosted model:

- **Baseline** (``LogisticRegression``): median-imputed, standard-scaled,
  with SMOTE oversampling applied to the *training* fold only.
- **XGBoost**: trained directly on raw features — NaN is handled natively
  via the library's learned split defaults — with class imbalance handled
  through ``scale_pos_weight`` instead of SMOTE, since SMOTE requires
  complete numeric input and would erase the missingness signal XGBoost
  can otherwise use.

Hyperparameter tuning (Optuna) and CLV regression are deferred until after
MVP per ``docs/roadmap.md``; this module always targets ``churn`` and uses
fixed, reasonable XGBoost defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data.cleaner import clean
from data.loader import load
from features.splits import build_time_splits, save_feature_names
from models.dataset import prepare_xy
from models.evaluate import (
    compute_classification_metrics,
    lift_at_k,
    revenue_at_risk,
    save_diagnostic_plots,
)
from models.preprocessing import apply_imputer, fit_imputer
from utils.config import ProjectConfig, load_config
from utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42
LIFT_K = 0.1


@dataclass
class BaselineModel:
    """A fitted logistic-regression baseline plus its preprocessing steps.

    Bundling the imputer/scaler with the model means the only way to score
    new data is through this object's own ``predict_proba``, which always
    replays the same fitted steps — there is no code path that could
    accidentally score with a stale or mismatched imputer.
    """

    imputer: SimpleImputer
    scaler: StandardScaler
    model: LogisticRegression

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = apply_imputer(self.imputer, X)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index,
        )
        return self.model.predict_proba(X_scaled)[:, 1]


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True,
    random_state: int = RANDOM_STATE,
) -> BaselineModel:
    """Fit the logistic-regression baseline.

    Fitting order is impute -> scale -> SMOTE, all fit on ``X_train`` only.
    SMOTE synthesizes new minority-class rows from the already
    imputed/scaled training fold; it can never affect validation/test data
    because it is not part of ``BaselineModel.predict_proba``.
    """
    imputer = fit_imputer(X_train)
    X_imputed = apply_imputer(imputer, X_train)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X_imputed.columns,
        index=X_imputed.index,
    )

    fit_X, fit_y = X_scaled, y_train
    if use_smote and y_train.nunique() > 1:
        minority_count = int(y_train.value_counts().min())
        if minority_count < 2:
            logger.warning(
                "train_baseline: minority class has only %d sample(s); skipping SMOTE",
                minority_count,
            )
        else:
            # SMOTE's default k_neighbors=5 raises if the minority class has
            # <=5 members. Clamping keeps this safe under any class balance
            # instead of crashing the moment a config change shifts it.
            k_neighbors = min(5, minority_count - 1)
            fit_X, fit_y = SMOTE(
                random_state=random_state, k_neighbors=k_neighbors
            ).fit_resample(X_scaled, y_train)

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(fit_X, fit_y)

    return BaselineModel(imputer=imputer, scaler=scaler, model=model)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    """Fit XGBoost directly on raw features (NaN handled natively)."""
    counts = y_train.value_counts()
    neg = int(counts.get(0, 0))
    pos = int(counts.get(1, 0))
    scale_pos_weight = neg / pos if pos else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def run_training(
    config: ProjectConfig | None = None,
    output_root: str | Path = "outputs",
) -> Path:
    """End-to-end: load, clean, split, train both models, evaluate, persist.

    Returns the timestamped artifact directory.
    """
    config = config or load_config()
    if config.modeling.target != "churn":
        raise NotImplementedError(
            "CLV regression is deferred until after MVP per docs/roadmap.md "
            f"section 4; got modeling.target={config.modeling.target!r}"
        )

    raw = load(config)
    cleaned = clean(raw, config)
    split = build_time_splits(cleaned, config)

    X_train, y_train = prepare_xy(split.train, target="churn")
    X_val, y_val = prepare_xy(split.val, target="churn")

    logger.info(
        "training baseline on %d rows x %d features", len(X_train), X_train.shape[1]
    )
    baseline = train_baseline(X_train, y_train, use_smote=config.modeling.use_smote)
    baseline_proba = baseline.predict_proba(X_val)

    logger.info(
        "training xgboost on %d rows x %d features", len(X_train), X_train.shape[1]
    )
    xgb_model = train_xgboost(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_val)[:, 1]

    # Revenue-at-risk must use a value that is actually known at scoring
    # time. ``clv`` (the label) is *future* revenue and would not exist yet
    # when scoring a live customer, so it can never be the "value" factor
    # here even for offline evaluation. ``spend_90d`` is the pre-snapshot
    # analogue — total revenue in the trailing 90 days — and comes straight
    # from ``X_val``, so it is guaranteed to be row-aligned with ``proba``.
    historical_value = X_val["spend_90d"].fillna(0.0).to_numpy()

    results: dict[str, dict] = {}
    for name, proba in (("baseline", baseline_proba), ("xgboost", xgb_proba)):
        metrics = compute_classification_metrics(y_val, proba)
        metrics["lift_at_10pct"] = lift_at_k(y_val, proba, k=LIFT_K)
        metrics["revenue_at_risk_total"] = float(
            revenue_at_risk(proba, historical_value).sum()
        )
        results[name] = metrics
        logger.info("%s val metrics: %s", name, metrics)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(baseline, output_dir / "baseline_model.joblib")
    joblib.dump(xgb_model, output_dir / "xgboost_model.joblib")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    params = {
        "baseline": {
            "use_smote": config.modeling.use_smote,
            "random_state": RANDOM_STATE,
        },
        "xgboost": xgb_model.get_params(),
    }
    with (output_dir / "params.json").open("w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, default=str)

    save_feature_names(split.train.features, output_dir / "feature_names.json")

    save_diagnostic_plots(
        y_val, baseline_proba, output_dir / "figures", prefix="baseline_"
    )
    save_diagnostic_plots(y_val, xgb_proba, output_dir / "figures", prefix="xgboost_")

    logger.info("Saved training artifacts to %s", output_dir)
    return output_dir
