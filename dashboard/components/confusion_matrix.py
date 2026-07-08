"""Confusion-matrix visualization.

No confusion-matrix image is saved to ``outputs/<timestamp>/figures/`` --
only the raw ``{tn, fp, fn, tp}`` counts in ``metrics.json`` -- so this
renders one from those counts rather than recomputing anything.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def confusion_matrix_figure(counts: dict[str, int]) -> Figure:
    """Build a labeled 2x2 heatmap figure from ``{tn, fp, fn, tp}`` counts."""
    matrix = np.array(
        [
            [counts["tn"], counts["fp"]],
            [counts["fn"], counts["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (tuned XGBoost)")
    fig.tight_layout()
    return fig
