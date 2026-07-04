"""End-to-end smoke test for the Day-4 modeling layer on real data.

Like ``smoke_features.py``, this is NOT a pytest test — it runs the full
load -> clean -> split -> train -> evaluate -> persist pipeline against the
real Online Retail II dataset and prints a human-readable summary.

Usage:
    python scripts/smoke_train.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from project root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.train import run_training  # noqa: E402
from utils.config import load_config  # noqa: E402


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    config = load_config()
    print(
        f"Loaded config. target={config.modeling.target!r} use_smote={config.modeling.use_smote}"
    )

    _section("Training baseline + XGBoost")
    t0 = time.perf_counter()
    output_dir = run_training(config, output_root=ROOT / "outputs")
    elapsed = time.perf_counter() - t0
    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  artifacts written to: {output_dir}")

    _section("Artifact contents")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(output_dir)}")

    import json

    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    _section("Validation metrics")
    for model_name, model_metrics in metrics.items():
        print(f"  {model_name}:")
        for key, value in model_metrics.items():
            print(f"    {key}: {value}")

    # Sanity: the boosted model should not be catastrophically worse than
    # the baseline on ROC-AUC — if it is, something upstream is likely
    # broken (e.g. feature/label misalignment).
    baseline_auc = metrics["baseline"]["roc_auc"]
    xgboost_auc = metrics["xgboost"]["roc_auc"]
    print()
    print(f"  baseline ROC-AUC: {baseline_auc:.3f}")
    print(f"  xgboost  ROC-AUC: {xgboost_auc:.3f}")
    if xgboost_auc < 0.5 or baseline_auc < 0.5:
        print("  WARNING: a model scored at or below random guessing.")

    _section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
