"""Tests for the Day-4 modeling layer.

Mirrors the property-based framing used in ``test_features.py``:

1. **Correctness** — outputs match hand-computed expectations on small
   synthetic data.
2. **Leakage prevention** — fitting (imputer, scaler, SMOTE) only ever
   touches training data; validation/test data is transformed with
   already-fitted objects, never refit.
3. **Business-metric correctness** — lift@k and revenue-at-risk match
   values worked out by hand.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from features.splits import FeatureLabelPair
from models.dataset import prepare_xy
from models.evaluate import (
    compute_classification_metrics,
    lift_at_k,
    revenue_at_risk,
)
from models.preprocessing import apply_imputer, fit_imputer
from models.cv import build_cv_report, run_cv_pipeline, run_time_cv
from models.train import train_baseline, train_xgboost

# ---------------------------------------------------------------------------
# dataset.prepare_xy
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_label_pair() -> FeatureLabelPair:
    features = pd.DataFrame(
        {
            "customer_id": [1.0, 2.0, 3.0, 4.0],
            "recency_days": [10, 20, 30, 40],
            "monetary": [100.0, 200.0, 300.0, 400.0],
        }
    )
    # Customer 4 has no label (e.g. entered the population right before the
    # snapshot) — FeatureLabelPair.joined leaves it as NaN by design.
    labels = pd.DataFrame(
        {
            "customer_id": [1.0, 2.0, 3.0],
            "churn": [0, 1, 0],
            "clv": [50.0, 0.0, 75.0],
        }
    )
    return FeatureLabelPair(features=features, labels=labels)


class TestPrepareXy:
    def test_drops_customers_with_no_label(self, feature_label_pair):
        X, y = prepare_xy(feature_label_pair, target="churn")
        assert len(X) == 3
        assert len(y) == 3

    def test_excludes_customer_id_from_features(self, feature_label_pair):
        X, _ = prepare_xy(feature_label_pair, target="churn")
        assert "customer_id" not in X.columns
        assert set(X.columns) == {"recency_days", "monetary"}

    def test_excludes_other_label_from_features(self, feature_label_pair):
        X, _ = prepare_xy(feature_label_pair, target="churn")
        assert "clv" not in X.columns
        assert "churn" not in X.columns

    def test_churn_target_is_int(self, feature_label_pair):
        _, y = prepare_xy(feature_label_pair, target="churn")
        assert y.tolist() == [0, 1, 0]

    def test_clv_target(self, feature_label_pair):
        X, y = prepare_xy(feature_label_pair, target="clv")
        assert len(X) == 3
        assert y.tolist() == [50.0, 0.0, 75.0]

    def test_invalid_target_raises(self, feature_label_pair):
        with pytest.raises(ValueError):
            prepare_xy(feature_label_pair, target="not_a_target")


# ---------------------------------------------------------------------------
# preprocessing.fit_imputer / apply_imputer
# ---------------------------------------------------------------------------


class TestImputer:
    def test_fills_nan_with_train_median(self):
        X_train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, 30.0]})
        imputer = fit_imputer(X_train)
        transformed = apply_imputer(imputer, X_train)
        assert transformed.loc[1, "a"] == 2.0  # median of [1, 3]

    def test_applies_train_statistic_to_val_not_vals_own(self):
        # Val's own median (100) is wildly different from train's (2). A
        # correct implementation fills val's NaN with train's median.
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        X_val = pd.DataFrame({"a": [np.nan, 100.0, 200.0]})

        imputer = fit_imputer(X_train)
        transformed_val = apply_imputer(imputer, X_val)

        assert transformed_val.loc[0, "a"] == 2.0

    def test_does_not_mutate_inputs(self):
        X_train = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        original = X_train.copy()

        imputer = fit_imputer(X_train)
        apply_imputer(imputer, X_train)

        pd.testing.assert_frame_equal(X_train, original)

    def test_preserves_columns_and_index(self):
        X_train = pd.DataFrame({"a": [1.0, np.nan]}, index=[5, 6])
        imputer = fit_imputer(X_train)
        transformed = apply_imputer(imputer, X_train)
        assert list(transformed.columns) == ["a"]
        assert list(transformed.index) == [5, 6]


# ---------------------------------------------------------------------------
# evaluate.compute_classification_metrics / lift_at_k / revenue_at_risk
# ---------------------------------------------------------------------------


class TestComputeClassificationMetrics:
    def test_perfect_predictions_score_maximally(self):
        y_true = [0, 0, 1, 1]
        y_proba = [0.0, 0.1, 0.9, 1.0]
        metrics = compute_classification_metrics(y_true, y_proba)
        assert metrics["roc_auc"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_confusion_matrix_counts(self):
        y_true = [0, 0, 1, 1]
        y_proba = [0.9, 0.1, 0.9, 0.1]  # one FP, one FN
        metrics = compute_classification_metrics(y_true, y_proba, threshold=0.5)
        cm = metrics["confusion_matrix"]
        assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}

    def test_returns_all_roadmap_metrics(self):
        y_true = [0, 1, 0, 1]
        y_proba = [0.2, 0.8, 0.3, 0.7]
        metrics = compute_classification_metrics(y_true, y_proba)
        expected_keys = (
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "brier_score",
            "confusion_matrix",
        )
        for key in expected_keys:
            assert key in metrics


class TestLiftAtK:
    def test_perfect_ranking_top_half(self):
        # Base rate = 0.5; top half ranked by proba is exactly the positives.
        y_true = [1, 1, 0, 0]
        y_proba = [0.9, 0.8, 0.2, 0.1]
        assert lift_at_k(y_true, y_proba, k=0.5) == pytest.approx(2.0)

    def test_random_ranking_no_lift(self):
        y_true = [1, 0, 1, 0]
        y_proba = [0.5, 0.5, 0.5, 0.5]
        # All tied — top-k by argsort is deterministic but not meaningfully
        # ranked; lift should equal 1.0 only when the top slice matches the
        # base rate, which this tied case with 2 positives/2 negatives does
        # for k=1.0 (the whole set).
        assert lift_at_k(y_true, y_proba, k=1.0) == pytest.approx(1.0)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            lift_at_k([1, 0], [0.5, 0.5], k=0.0)
        with pytest.raises(ValueError):
            lift_at_k([1, 0], [0.5, 0.5], k=1.5)


class TestRevenueAtRisk:
    def test_elementwise_product(self):
        churn_proba = np.array([0.5, 0.2, 0.9])
        predicted_value = np.array([100.0, 200.0, 50.0])
        result = revenue_at_risk(churn_proba, predicted_value)
        np.testing.assert_allclose(result, [50.0, 40.0, 45.0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            revenue_at_risk(np.array([0.5, 0.5]), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# train.train_baseline / train.train_xgboost
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_training_data():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(
        {
            "recency_days": rng.uniform(0, 100, n),
            "monetary": rng.uniform(10, 1000, n),
            "frequency": rng.uniform(1, 20, n),
        }
    )
    # Introduce some missingness, and a label correlated with recency so the
    # models have something real to learn.
    X.loc[rng.choice(n, size=10, replace=False), "monetary"] = np.nan
    y = pd.Series((X["recency_days"] > 50).astype(int).values)
    return X, y


class TestTrainBaseline:
    def test_returns_fitted_pipeline_components(self, synthetic_training_data):
        X_train, y_train = synthetic_training_data
        result = train_baseline(X_train, y_train)
        assert hasattr(result.model, "predict_proba")
        proba = result.predict_proba(X_train)
        assert proba.shape == (len(X_train),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_does_not_mutate_training_data(self, synthetic_training_data):
        X_train, y_train = synthetic_training_data
        original_X = X_train.copy()
        original_y = y_train.copy()

        train_baseline(X_train, y_train)

        pd.testing.assert_frame_equal(X_train, original_X)
        pd.testing.assert_series_equal(y_train, original_y)

    def test_handles_nan_via_imputation(self, synthetic_training_data):
        X_train, y_train = synthetic_training_data
        assert X_train.isna().any().any()  # sanity: fixture has NaN
        result = train_baseline(X_train, y_train)
        # Predicting on data with NaN must not raise — the fitted imputer
        # is reused internally.
        proba = result.predict_proba(X_train)
        assert not np.isnan(proba).any()

    def test_scaler_fit_on_train_only(self, synthetic_training_data):
        X_train, y_train = synthetic_training_data
        result = train_baseline(X_train, y_train)
        X_train_imputed = apply_imputer(result.imputer, X_train)
        np.testing.assert_allclose(
            result.scaler.mean_, X_train_imputed.mean().to_numpy()
        )

    def test_predict_does_not_refit_scaler_on_val_distribution(
        self, synthetic_training_data
    ):
        X_train, y_train = synthetic_training_data
        result = train_baseline(X_train, y_train)
        original_mean = result.scaler.mean_.copy()

        # A validation set with a wildly different distribution must not
        # change the already-fitted scaler when scored.
        X_val = X_train.copy() * 100
        result.predict_proba(X_val)

        np.testing.assert_array_equal(result.scaler.mean_, original_mean)

    def test_smote_survives_tiny_minority_class(self):
        # Minority class of 3 is below SMOTE's default k_neighbors=5
        # requirement — this must be handled, not crash.
        rng = np.random.RandomState(0)
        n = 50
        X_train = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
        y_train = pd.Series([0] * 47 + [1] * 3)

        result = train_baseline(X_train, y_train, use_smote=True)
        proba = result.predict_proba(X_train)
        assert proba.shape == (n,)

    def test_skips_smote_with_single_minority_sample(self):
        n = 50
        X_train = pd.DataFrame({"a": np.arange(n, dtype=float)})
        y_train = pd.Series([0] * (n - 1) + [1])

        result = train_baseline(X_train, y_train, use_smote=True)
        proba = result.predict_proba(X_train)
        assert proba.shape == (n,)


class TestTrainXgboost:
    def test_trains_on_raw_features_with_nan(self, synthetic_training_data):
        X_train, y_train = synthetic_training_data
        model = train_xgboost(X_train, y_train)
        proba = model.predict_proba(X_train)[:, 1]
        assert proba.shape == (len(X_train),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_does_not_mutate_training_data(self, synthetic_training_data):
        X_train, y_train = synthetic_training_data
        original_X = X_train.copy()

        train_xgboost(X_train, y_train)

        pd.testing.assert_frame_equal(X_train, original_X)


# ---------------------------------------------------------------------------
# cv.run_time_cv / build_cv_report
# ---------------------------------------------------------------------------


@pytest.fixture
def cv_fixtures():
    """Create a deterministic 40-row dataset for CV tests.

    With 40 rows and TimeSeriesSplit(n_splits=4):
      - test_size = 40 / (4+1) = 8
      - fold 1: train=8, val=8
      - fold 2: train=16, val=8
      - fold 3: train=24, val=8
      - fold 4: train=32, val=8

    Even the smallest training fold has 8 rows, which with roughly 50/50
    class balance guarantees at least 2 minority samples per fold — enough
    for SMOTE to run and for logistic regression to converge.
    """
    rng = np.random.RandomState(0)
    n = 40
    X = pd.DataFrame(
        {
            "recency_days": rng.uniform(0, 100, n),
            "monetary": rng.uniform(10, 1000, n),
        }
    )
    # y is correlated with recency so the model has a real signal to learn.
    y = pd.Series((X["recency_days"] > 50).astype(int).values)
    return X, y


class TestRunTimeCv:
    def test_returns_results_for_both_models(self, cv_fixtures):
        X, y = cv_fixtures
        results = run_time_cv(X, y, n_folds=3)
        assert "baseline" in results
        assert "xgboost" in results

    def test_correct_number_of_folds(self, cv_fixtures):
        X, y = cv_fixtures
        n_folds = 4
        results = run_time_cv(X, y, n_folds=n_folds)
        assert len(results["baseline"]["fold_metrics"]) == n_folds
        assert len(results["xgboost"]["fold_metrics"]) == n_folds

    def test_fold_indices_are_time_ordered(self, cv_fixtures):
        X, y = cv_fixtures
        n_folds = 3

        results = run_time_cv(X, y, n_folds=n_folds)
        # TimeSeriesSplit always expands: train sizes should be strictly
        # increasing across folds.
        train_sizes = [fm["n_train"] for fm in results["baseline"]["fold_metrics"]]
        assert train_sizes == sorted(train_sizes), "TimeSeriesSplit must expand"

    def test_aggregate_mean_and_std_present(self, cv_fixtures):
        X, y = cv_fixtures
        results = run_time_cv(X, y, n_folds=3)
        for name in ("baseline", "xgboost"):
            assert "mean" in results[name]
            assert "std" in results[name]
            assert "roc_auc" in results[name]["mean"]
            assert "pr_auc" in results[name]["mean"]

    def test_roc_auc_and_pr_auc_in_fold_metrics(self, cv_fixtures):
        X, y = cv_fixtures
        results = run_time_cv(X, y, n_folds=3)
        for name in ("baseline", "xgboost"):
            for fm in results[name]["fold_metrics"]:
                assert "roc_auc" in fm
                assert "pr_auc" in fm
                assert 0 <= fm["roc_auc"] <= 1
                assert 0 <= fm["pr_auc"] <= 1

    def test_mean_in_range_of_fold_values(self, cv_fixtures):
        X, y = cv_fixtures
        results = run_time_cv(X, y, n_folds=3)
        for name in ("baseline", "xgboost"):
            for metric in ("roc_auc", "pr_auc"):
                mean = results[name]["mean"][metric]
                std = results[name]["std"][metric]
                fold_values = [fm[metric] for fm in results[name]["fold_metrics"]]
                assert mean >= min(fold_values) - 1e-6
                assert mean <= max(fold_values) + 1e-6
                assert std >= 0

    def test_scale_pos_weight_computed_if_not_supplied(self, cv_fixtures):
        X, y = cv_fixtures
        results = run_time_cv(X, y, n_folds=3, scale_pos_weight=None)
        # Should not raise; scale_pos_weight was derived from the split.
        assert "baseline" in results


class TestBuildCvReport:
    def test_returns_dataframe_with_correct_columns(self, cv_fixtures):
        X, y = cv_fixtures
        cv_results = run_time_cv(X, y, n_folds=3)
        df = build_cv_report(cv_results)
        expected_cols = {
            "model",
            "metric",
            "single_split",
            "cv_mean",
            "cv_std",
            "delta (ss - cv)",
        }
        assert set(df.columns) == expected_cols

    def test_one_row_per_model_metric_pair(self, cv_fixtures):
        X, y = cv_fixtures
        cv_results = run_time_cv(X, y, n_folds=3)
        df = build_cv_report(cv_results)
        assert len(df) == 4  # 2 models x 2 metrics

    def test_single_split_included_when_provided(self, cv_fixtures):
        X, y = cv_fixtures
        cv_results = run_time_cv(X, y, n_folds=3)
        single_split = {"baseline": {"roc_auc": 0.80, "pr_auc": 0.85}}
        df = build_cv_report(cv_results, single_split_metrics=single_split)
        row = df[(df["model"] == "baseline") & (df["metric"] == "roc_auc")]
        assert row["single_split"].values[0] == 0.80

    def test_delta_computed_when_single_split_provided(self, cv_fixtures):
        X, y = cv_fixtures
        cv_results = run_time_cv(X, y, n_folds=3)
        single_split = {"baseline": {"roc_auc": 0.80, "pr_auc": 0.85}}
        df = build_cv_report(cv_results, single_split_metrics=single_split)
        row = df[(df["model"] == "baseline") & (df["metric"] == "roc_auc")]
        delta = row["delta (ss - cv)"].values[0]
        assert delta is not None
        # delta = single_split - cv_mean
        assert abs(delta - (0.80 - row["cv_mean"].values[0])) < 1e-6

    def test_delta_is_nan_when_single_split_not_provided(self, cv_fixtures):
        X, y = cv_fixtures
        cv_results = run_time_cv(X, y, n_folds=3)
        df = build_cv_report(cv_results, single_split_metrics=None)
        assert df["delta (ss - cv)"].isna().all()

    def test_smoke_run_cv_pipeline(self, cv_fixtures, tmp_path):
        X, y = cv_fixtures
        cv_results, comparison_df, output_path = run_cv_pipeline(
            X, y, n_folds=3, output_dir=tmp_path, compare_single_split=False
        )
        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "cv_results" in data
        assert "comparison" in data
