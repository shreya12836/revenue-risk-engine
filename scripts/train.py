"""End-to-end training pipeline: tune, retrain, evaluate, explain, persist.

Like ``smoke_train.py``, this is NOT a pytest test — it runs the full
tune -> retrain -> evaluate -> explain -> persist -> verify pipeline against
the real Online Retail II dataset and prints a human-readable summary. The
final step reloads the just-saved artifacts through ``ChurnPredictor`` (the
same shared inference module the future FastAPI endpoint will use) as an
integration check that training output and inference input actually agree.

Usage:
    python scripts/train.py --config configs/online_retail_ii.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.cleaner import clean  # noqa: E402
from data.loader import load  # noqa: E402
from features.splits import build_time_splits, save_feature_names  # noqa: E402
from models.dataset import prepare_xy  # noqa: E402
from models.evaluate import (  # noqa: E402
    compute_classification_metrics,
    lift_at_k,
    revenue_at_risk,
    save_diagnostic_plots,
)
from models.explain import (  # noqa: E402
    build_feature_importance_table,
    compute_shap_values,
    save_shap_summary_plot,
    save_shap_waterfall_plot,
)
from models.predict import ChurnPredictor, build_feature_schema, save_feature_schema  # noqa: E402
from models.train import LIFT_K, train_baseline, train_xgboost  # noqa: E402
from models.tuning import train_xgboost_with_params, tune_xgboost  # noqa: E402
from models.versioning import build_metadata, save_model_version  # noqa: E402
from utils.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _evaluate(proba, y_true, historical_value) -> dict:
    metrics = compute_classification_metrics(y_true, proba)
    metrics["lift_at_10pct"] = lift_at_k(y_true, proba, k=LIFT_K)
    metrics["revenue_at_risk_total"] = float(
        revenue_at_risk(proba, historical_value).sum()
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the project YAML config (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    tuning_cfg = config.modeling.hyperparameter_tuning
    print(
        f"Loaded config. n_trials={tuning_cfg.n_trials} timeout={tuning_cfg.timeout}s "
        f"cv_folds={config.modeling.cv_folds}"
    )

    _section("Load, clean, split")
    raw = load(config)
    cleaned = clean(raw, config)
    split = build_time_splits(cleaned, config)

    X_train, y_train = prepare_xy(split.train, target="churn")
    X_val, y_val = prepare_xy(split.val, target="churn")
    X_test, y_test = prepare_xy(split.test, target="churn")
    # Mirrors prepare_xy's own filter exactly, so customer_ids_test stays
    # row-aligned with X_test/y_test (prepare_xy itself drops customer_id
    # from X since it's a join key, not a model input).
    customer_ids_test = (
        split.test.joined.dropna(subset=["churn"])["customer_id"].reset_index(drop=True)
    )

    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    _section("Optuna hyperparameter search (PR-AUC objective)")
    t0 = time.perf_counter()
    study = tune_xgboost(
        X_train,
        y_train,
        n_trials=tuning_cfg.n_trials,
        timeout=tuning_cfg.timeout,
        n_folds=config.modeling.cv_folds,
    )
    print(f"  elapsed: {time.perf_counter() - t0:.2f}s")
    print(f"  trials run: {len(study.trials)}")
    print(f"  best CV PR-AUC: {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")

    _section("Retrain tuned XGBoost on train+val")
    tuned_model = train_xgboost_with_params(X_trainval, y_trainval, study.best_params)

    _section("Train baseline + default XGBoost for comparison")
    baseline = train_baseline(X_train, y_train, use_smote=config.modeling.use_smote)
    default_xgb = train_xgboost(X_train, y_train)

    _section("Evaluate all three models on the held-out test split")
    historical_value = X_test["spend_90d"].fillna(0.0).to_numpy()
    tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
    results = {
        "baseline": _evaluate(baseline.predict_proba(X_test), y_test, historical_value),
        "xgboost_default": _evaluate(
            default_xgb.predict_proba(X_test)[:, 1], y_test, historical_value
        ),
        "xgboost_tuned": _evaluate(tuned_proba, y_test, historical_value),
    }
    for name, metrics in results.items():
        print(
            f"  {name}: PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f}"
        )

    _section("SHAP explainability on tuned model")
    shap_values = compute_shap_values(tuned_model, X_test)
    feature_importance = build_feature_importance_table(
        tuned_model, shap_values, list(X_test.columns)
    )
    highest_risk_idx = int(pd.Series(tuned_proba).idxmax())
    highest_risk_customer_id = int(customer_ids_test.iloc[highest_risk_idx])
    print("  top 3 features by mean|SHAP|:")
    for _, row in feature_importance.head(3).iterrows():
        print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
    print(f"  explaining highest-risk customer_id={highest_risk_customer_id}")

    _section("Persist artifacts")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "outputs" / timestamp
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(
        tuned_model,
        feature_count=X_test.shape[1],
        split_sizes={"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        study=study,
        cv_folds=config.modeling.cv_folds,
        version="v1",
    )
    save_model_version(tuned_model, metadata, output_dir, version="v1")

    schema = build_feature_schema(X_trainval, version="v1")
    save_feature_schema(schema, output_dir / "feature_schema.json")
    save_feature_names(split.train.features, output_dir / "feature_names.json")

    (output_dir / "best_params.json").write_text(
        json.dumps(study.best_params, indent=2), encoding="utf-8"
    )
    # Nice-to-have, not a hard requirement: best_params.json alone is
    # sufficient to reproduce the tuned model.
    joblib.dump(study, output_dir / "optuna_study.pkl")

    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    save_shap_summary_plot(shap_values, X_test, figures_dir / "shap_summary.png")
    save_shap_waterfall_plot(
        shap_values,
        X_test,
        customer_ids_test,
        customer_id=highest_risk_customer_id,
        output_path=figures_dir / f"shap_waterfall_{highest_risk_customer_id}.png",
    )
    save_diagnostic_plots(y_test, tuned_proba, figures_dir, prefix="xgboost_tuned_")

    print(f"  artifacts written to: {output_dir}")

    _section("Verify: reload persisted artifacts through ChurnPredictor")
    predictor = ChurnPredictor.from_artifacts(output_dir, version="v1")
    reloaded_proba, reloaded_shap = predictor.predict_with_shap(X_test)
    np.testing.assert_allclose(reloaded_proba, tuned_proba)
    assert reloaded_shap.values.shape == (len(X_test), X_test.shape[1])
    print(
        "  OK: model_v1.joblib + feature_schema.json round-trip through "
        "ChurnPredictor and reproduce the in-memory predictions exactly."
    )

    _section("Summary")
    print(f"  baseline        PR-AUC: {results['baseline']['pr_auc']:.4f}")
    print(f"  xgboost default PR-AUC: {results['xgboost_default']['pr_auc']:.4f}")
    print(f"  xgboost tuned   PR-AUC: {results['xgboost_tuned']['pr_auc']:.4f}")
    print(f"  git commit: {metadata['git_commit']}")
    print(f"  output dir: {output_dir}")

    _section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
