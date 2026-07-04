"""Model evaluation: classification metrics, lift, and revenue-at-risk.

Metrics tracked here follow the roadmap's Section 4 list directly, plus
the business-facing lift@k and revenue-at-risk figures that translate a
churn score into "who should retention actually call first."
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | dict[str, int]]:
    """Return the roadmap's core churn-classification metrics.

    ``y_proba`` is the predicted probability of the positive (churn) class.
    ``threshold`` only affects the metrics that need a hard label
    (precision/recall/F1/confusion matrix) — ROC-AUC, PR-AUC, and Brier
    score are threshold-independent by definition.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def lift_at_k(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    k: float = 0.1,
) -> float:
    """Lift of the top-``k`` fraction of customers ranked by predicted risk.

    Lift = (positive rate among the top-k highest-risk customers) /
    (overall positive rate). A lift of 2.0 at k=0.1 means the top decile
    contains churners at twice the base rate — the number a retention team
    cares about when deciding who to target first.
    """
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}")

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    top_n = max(1, int(np.ceil(n * k)))

    order = np.argsort(-y_proba)
    top_indices = order[:top_n]

    base_rate = y_true.mean()
    if base_rate == 0:
        return float("nan")

    top_rate = y_true[top_indices].mean()
    return float(top_rate / base_rate)


def revenue_at_risk(
    churn_proba: pd.Series | np.ndarray,
    predicted_value: pd.Series | np.ndarray,
) -> np.ndarray:
    """Per-customer revenue-at-risk: churn probability x predicted value.

    Per the roadmap's business-framing assumption, revenue-at-risk is the
    expected revenue loss if a customer churns, weighted by how likely
    that is.
    """
    churn_proba = np.asarray(churn_proba)
    predicted_value = np.asarray(predicted_value)
    if churn_proba.shape != predicted_value.shape:
        raise ValueError(
            f"shape mismatch: churn_proba {churn_proba.shape} vs "
            f"predicted_value {predicted_value.shape}"
        )
    return churn_proba * predicted_value


def save_diagnostic_plots(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    output_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """Save ROC, precision-recall, and calibration plots as PNGs.

    Returns the list of written file paths.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    path = output_dir / f"{prefix}roc_curve.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(
        recall,
        precision,
        label=f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    path = output_dir / f"{prefix}pr_curve.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    path = output_dir / f"{prefix}calibration_curve.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    return written
