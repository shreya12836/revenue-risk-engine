"""Time-based cross-validation for churn classification.

**Design decisions:**

1. **TimeSeriesSplit** is used instead of random K-fold because transaction
   data is temporally ordered — the model must never train on data that
   occurs after the validation period it is being evaluated on.

2. **SMOTE is applied *inside* each CV fold** (on the training portion only).
   Applying SMOTE before splitting would leak synthetic samples from the
   validation fold back into training, inflating metrics. The
   ``imblearn.pipeline.Pipeline`` wrapper makes this discipline automatic:
   ``impute`` and ``scale`` are fit on train, then ``SMOTE`` is applied to the
   already-imputed/scaled train split. Validation data is transformed by the
   fitted ``impute`` and ``scale`` steps but *never* enters SMOTE.

3. **Baseline** uses an imblearn ``Pipeline`` wrapping
   ``SimpleImputer → StandardScaler → SMOTE`` before ``LogisticRegression``.
   This is the pipeline used in Day 4 for the single-split baseline, so the
   CV metrics are directly comparable to the existing val-set numbers.

4. **XGBoost** is trained directly on raw features (NaN handled natively) with
   ``scale_pos_weight`` for class imbalance — no SMOTE. Default hyperparameters
   are used; Optuna tuning happens in Day 6.

5. **Metrics reported per fold**: ROC-AUC and PR-AUC (the two metrics most
   robust to class imbalance). Aggregate mean ± std across folds documents
   variance, which is the Day 5 deliverable for comparing single-split vs CV.

6. **Comparison table**: The single-split val metrics from ``run_training``'s
   last run are read from the latest ``outputs/<timestamp>/metrics.json`` and
   placed alongside the CV averages so the variance introduced by using a
   single fixed val split is visible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Core CV routine
# ---------------------------------------------------------------------------


def _build_baseline_pipeline(
    minority_count: int | None = None,
) -> ImbPipeline:
    """Build the imblearn pipeline used for the logistic-regression baseline.

    Steps: impute → scale → SMOTE → LogisticRegression
    All steps are *inside* the pipeline; sklearn's cross_val_predict / scorer
    will automatically apply impute/scale to the validation fold without
    refitting, and will only call SMOTE on the training fold.

    ``minority_count`` lets callers clamp SMOTE's ``k_neighbors`` to avoid the
    default ``k_neighbors=5`` raising when the minority class has fewer than
    5+1 members (SMOTE adds 1 internally for self-neighbor exclusion).
    When ``minority_count`` is ``None`` SMOTE uses its default ``k_neighbors=5``.
    When ``minority_count < 2`` the pipeline omits SMOTE entirely (fallback to
    plain logistic regression — still valid; SMOTE is an enhancement).
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]
    # Only add SMOTE when minority_count >= 2 (enough samples for k_neighbors >= 1).
    if minority_count is not None and minority_count >= 2:
        # Clamp k_neighbors so SMOTE never receives k_neighbors=0 (which raises).
        effective_k = max(1, min(5, minority_count - 1))
        steps.insert(
            2, ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=effective_k))
        )

    return ImbPipeline(steps)


def _build_xgboost() -> XGBClassifier:
    """Build XGBoost with default (pre-tuning) hyperparameters.

    ``scale_pos_weight`` is computed from the training fold's class counts at
    call time, so it adapts correctly to whatever class ratio appears in each
    CV fold — no global constant is baked in.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )


def run_time_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    scale_pos_weight: float | None = None,
) -> dict:
    """Run time-based CV and return per-fold + aggregate metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no ``customer_id`` column).
    y : pd.Series
        Binary churn labels.
    n_folds : int
        Number of CV folds (passed to ``TimeSeriesSplit``).
    scale_pos_weight : float, optional
        Pre-computed ``scale_pos_weight`` for XGBoost (computed from the
        full training split if not supplied).

    Returns
    -------
    dict
        ``{"baseline": {"fold_metrics": [...], "mean": {...}, "std": {...}},
           "xgboost": {"fold_metrics": [...], "mean": {...}, "std": {...}}}``
        Each ``fold_metrics`` entry has ``{"roc_auc": float, "pr_auc": float,
        "n_train": int, "n_val": int}``.
    """
    xgb_model = _build_xgboost()

    if scale_pos_weight is None:
        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        scale_pos_weight = neg / pos if pos else 1.0
    xgb_model.set_params(scale_pos_weight=scale_pos_weight)

    tscv = TimeSeriesSplit(n_splits=n_folds)

    results: dict[str, dict] = {
        "baseline": {"fold_metrics": [], "mean": {}, "std": {}},
        "xgboost": {"fold_metrics": [], "mean": {}, "std": {}},
    }

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        y_train_fold = y.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)
        y_val_fold = y.iloc[val_idx].reset_index(drop=True)

        # Compute minority count per fold for the k_neighbors guard.
        minority_count: int | None = None
        n_classes = y_train_fold.nunique()
        if n_classes > 1:
            minority_count = int(y_train_fold.value_counts().min())

        # ---------- Baseline (imblearn Pipeline handles SMOTE-on-train-only) ----------
        # When n_classes == 1, SMOTE and logistic regression both fail. Skip this
        # fold rather than report a broken metric — it's a data artifact of tiny
        # CV folds, not a model failure.
        if n_classes == 1:
            logger.warning(
                "Fold %d: training fold has only 1 class; skipping this fold",
                fold_idx + 1,
            )
            continue
        baseline_pipe_fold = _build_baseline_pipeline(minority_count=minority_count)
        baseline_pipe_fold.fit(X_train_fold, y_train_fold)
        baseline_proba = baseline_pipe_fold.predict_proba(X_val_fold)[:, 1]
        baseline_roc = float(roc_auc_score(y_val_fold, baseline_proba))
        baseline_pr = float(average_precision_score(y_val_fold, baseline_proba))
        results["baseline"]["fold_metrics"].append(
            {
                "fold": fold_idx + 1,
                "roc_auc": baseline_roc,
                "pr_auc": baseline_pr,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )

        # ---------- XGBoost (no SMOTE; scale_pos_weight from full split) ----------
        xgb_fold = _build_xgboost()
        xgb_fold.set_params(scale_pos_weight=scale_pos_weight)
        xgb_fold.fit(X_train_fold, y_train_fold)
        xgb_proba = xgb_fold.predict_proba(X_val_fold)[:, 1]
        xgb_roc = float(roc_auc_score(y_val_fold, xgb_proba))
        xgb_pr = float(average_precision_score(y_val_fold, xgb_proba))
        results["xgboost"]["fold_metrics"].append(
            {
                "fold": fold_idx + 1,
                "roc_auc": xgb_roc,
                "pr_auc": xgb_pr,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )

        logger.info(
            "Fold %d/%d: train=%d val=%d | baseline ROC=%.3f PR=%.3f | "
            "XGBoost ROC=%.3f PR=%.3f",
            fold_idx + 1,
            n_folds,
            len(train_idx),
            len(val_idx),
            baseline_roc,
            baseline_pr,
            xgb_roc,
            xgb_pr,
        )

    # Aggregate mean ± std across folds
    for name in ("baseline", "xgboost"):
        for metric in ("roc_auc", "pr_auc"):
            values = [fm[metric] for fm in results[name]["fold_metrics"]]
            results[name]["mean"][metric] = float(np.mean(values))
            results[name]["std"][metric] = float(np.std(values))

    return results


# ---------------------------------------------------------------------------
# Comparison: single-split vs CV
# ---------------------------------------------------------------------------


def _load_latest_val_metrics(output_root: str | Path = "outputs") -> dict | None:
    """Read the most recent ``metrics.json`` from ``outputs/<timestamp>/``."""
    output_root = Path(output_root)
    if not output_root.exists():
        return None
    timestamps = [
        (p, p.stat().st_mtime)
        for p in output_root.iterdir()
        if p.is_dir() and (p / "metrics.json").exists()
    ]
    if not timestamps:
        return None
    latest = max(timestamps, key=lambda x: x[1])[0]
    return json.loads((latest / "metrics.json").read_text(encoding="utf-8"))


def build_cv_report(
    cv_results: dict,
    single_split_metrics: dict | None = None,
) -> pd.DataFrame:
    """Build a human-readable comparison table of CV vs single-split metrics.

    Parameters
    ----------
    cv_results : dict
        Output of ``run_time_cv``.
    single_split_metrics : dict, optional
        ``{"baseline": {...}, "xgboost": {...}}`` from the latest
        ``metrics.json``. If not supplied, the comparison columns will be
        blank but the CV table is still returned.

    Returns
    -------
    pd.DataFrame
        One row per (model, metric) pair with columns:
        ``model``, ``metric``, ``single_split``, ``cv_mean``, ``cv_std``, ``delta``
    """
    rows = []
    for name in ("baseline", "xgboost"):
        for metric in ("roc_auc", "pr_auc"):
            cv_mean = cv_results[name]["mean"][metric]
            cv_std = cv_results[name]["std"][metric]
            single = (
                single_split_metrics.get(name, {}).get(metric)
                if single_split_metrics
                else None
            )
            delta = (single - cv_mean) if single is not None else None
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "single_split": single,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "delta (ss - cv)": delta,
                }
            )
    return pd.DataFrame(rows)


def save_cv_results(
    cv_results: dict,
    comparison_df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Write CV results and comparison table to ``output_dir/cv_results.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cv_results": cv_results,
        "comparison": comparison_df.to_dict(orient="records"),
    }
    path = output_dir / "cv_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved CV results to %s", path)
    return path


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_cv_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = 5,
    output_dir: str | Path = "outputs",
    compare_single_split: bool = True,
) -> tuple[dict, pd.DataFrame, Path]:
    """Run the full CV pipeline and persist results.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        The training split's feature matrix and labels.
    n_folds : int
        Number of TimeSeriesSplit folds.
    output_dir : str | Path
        Root ``outputs/`` directory (timestamped sub-folder is created).
    compare_single_split : bool
        If True, load the latest ``metrics.json`` and include single-split
        comparison in the report.

    Returns
    -------
    tuple[dict, pd.DataFrame, Path]
        (cv_results, comparison_df, output_path)
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Compute scale_pos_weight from the full training split for XGBoost
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos else 1.0

    cv_results = run_time_cv(
        X_train, y_train, n_folds=n_folds, scale_pos_weight=scale_pos_weight
    )

    single_split_metrics = None
    if compare_single_split:
        single_split_metrics = _load_latest_val_metrics(output_dir)
        if single_split_metrics:
            logger.info(
                "Loaded single-split val metrics from latest run for comparison"
            )

    comparison_df = build_cv_report(cv_results, single_split_metrics)

    output_path = save_cv_results(cv_results, comparison_df, run_dir)

    return cv_results, comparison_df, output_path
