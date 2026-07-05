# Revenue Risk Engine

Production-oriented customer churn and revenue-at-risk prediction system.

The project is currently in its foundation phase. The first working contracts are
the project configuration, shared utilities, and tests that keep the scaffold
honest while the data, modeling, API, and dashboard layers are built out.

## Planned Capabilities

- Load and clean Online Retail II transaction data.
- Build customer-level churn and revenue features.
- Train and evaluate churn / CLV models with XGBoost and LightGBM.
- Explain predictions with SHAP.
- Serve predictions through FastAPI.
- Explore customer risk through a Streamlit dashboard.

## Repository Layout

```text
configs/              YAML configuration files
scripts/              Smoke tests and standalone utilities
src/api/              FastAPI application package
src/data/             Data loading, cleaning, and feature engineering
src/features/         Feature engineering (RFM, rolling, trend, labels, splits)
src/models/           Training, evaluation, and inference
src/utils/            Shared utilities
tests/                Pytest test suite
```

## Setup

```bash
pip install -e ".[dev]"
```

## Common Commands

```bash
make test
make lint
make format
make run-api
make run-dashboard
```

`make run-api` and `make run-dashboard` are placeholders until the FastAPI app
and Streamlit dashboard are implemented.

## Current Configuration

The default project config lives at `configs/online_retail_ii.yaml`. It defines:

- Dataset source and local path.
- Raw schema column names.
- Cleaning rules.
- Churn and CLV feature windows.
- Model families and tuning settings.
- Output directories for model and reporting artifacts.

## Near-Term Roadmap

1. ~~Stabilize scaffold with tests, docs, and lint-clean utilities.~~ ✅
2. ~~Add config-driven data loading and cleaning.~~ ✅
3. ~~Build customer feature generation and labels.~~ ✅
4. ~~Add baseline churn model training and evaluation.~~ ✅
5. ~~Add Optuna tuning, SHAP explainability, and versioned model artifacts.~~ ✅
6. Expose the versioned model through a FastAPI prediction endpoint
   (`ChurnPredictor` — see Day 6 — is the inference layer this will wrap).
7. Add dashboard views for customer risk and revenue exposure.

## Feature Engineering (Day 3)

The feature layer is leakage-safe by construction: every feature function
raises a loud `ValueError` if it receives any transaction dated after the
snapshot date. Snapshot-date alignment is verified by a dedicated
`TestLeakageIsImpossible` integration test.

**Feature modules:**

| Module | What it computes |
|--------|-----------------|
| `features.rfm` | Recency, frequency, monetary |
| `features.rolling` | 30 / 60 / 90-day windowed aggregates |
| `features.customer_stats` | Tenure, AOV, basket size, distinct products |
| `features.trend` | Spend slope, transaction-count slope, inter-purchase gap |
| `features.labels` | Churn (binary) and CLV (future revenue) labels |
| `features.splits` | Time-aware train / val / test split construction |

Every label window is also guarded by `assert_sufficient_future_window`: if
a snapshot sits too close to the end of the available data, building labels
raises instead of silently mislabeling censored customers as churned.

**Verified on real data** (Online Retail II, both sheets, 1,067,371 rows):

| Split | Snapshot   | Customers | Features | Churn rate |
|-------|------------|-----------|----------|------------|
| train | 2010-06-01 | 2,577     | 33       | 51.3%      |
| val   | 2010-12-01 | 4,096     | 33       | 67.4%      |
| test  | 2011-09-01 | 5,053     | 33       | 57.5%      |

The test snapshot is 2011-09-01, not the data's tail end (2011-12-09) —
close to the end, and there's not enough runway left for a 90-day churn/CLV
window to observe genuine repeat purchases, so labels quietly collapse
toward "everyone churned." `assert_sufficient_future_window` now raises the
moment a config change would do that again.

Run the smoke test to reproduce: `python scripts/smoke_features.py`

## Modeling (Day 4)

Two churn classifiers are trained per run and compared on the same
time-aware validation split — a simple baseline before any boosted model,
per the project roadmap:

| Model | Preprocessing | Imbalance handling |
|-------|---------------|---------------------|
| Logistic Regression (baseline) | Median imputation + standard scaling, fit on train only | SMOTE, applied to the training fold only |
| XGBoost | None — raw features, `NaN` handled natively via learned split defaults | `scale_pos_weight` |

Both the imputer and scaler are fit exclusively on the training split and
reused (never refit) on validation/test data — the same fit-on-train,
apply-everywhere discipline used by `remove_outliers` in the cleaning
layer. XGBoost never sees imputed or SMOTE-resampled data: SMOTE requires
complete numeric input, which would erase the missingness signal the
model can otherwise learn from directly.

**Validation metrics** (train snapshot 2010-06-01, val snapshot
2010-12-01, `churn_window_days=90`):

| Metric | Logistic Regression | XGBoost |
|--------|---------------------|---------|
| ROC-AUC | 0.780 | 0.755 |
| PR-AUC | 0.859 | 0.830 |
| Precision | 0.854 | 0.828 |
| Recall | 0.637 | 0.640 |
| F1 | 0.730 | 0.722 |
| Brier score | 0.238 | 0.223 |
| Lift @ top 10% | 1.36x | 1.27x |
| Revenue-at-risk (val total) | $268,920 | $340,356 |

Revenue-at-risk is `churn_probability x spend_90d` — trailing 90-day
spend, the same window length as the churn label but looking backward
instead of forward, so it is a value actually known at scoring time. Using
the *future* revenue label here would be leakage-adjacent and impossible
to reproduce for a live customer.

XGBoost defaults underperform logistic regression on this dataset
(PR-AUC 0.830 vs 0.859, ROC-AUC 0.755 vs 0.780). This is expected: with
only ~2,600 training rows and 33 features, tree-based models need tuning
to beat a well-regularized linear baseline. Hyperparameter optimization
(Optuna) is deferred to v2 per the roadmap — this comparison is the
honest pre-tuning starting point, not a final result, and is exactly what
tuning will be measured against.

**Artifacts saved per run** to `outputs/<UTC-timestamp>/`: both fitted
models (`.joblib`), `metrics.json`, `params.json`, `feature_names.json`,
and ROC / precision-recall / calibration-curve plots per model.

Run the smoke test to reproduce: `python scripts/smoke_train.py`

## Hyperparameter Tuning & Explainability (Day 6)

XGBoost's defaults were never tuned in Day 4 — hyperparameter search was
explicitly deferred until this step. An Optuna search (50 trials, 10-minute
timeout, both read from `configs/online_retail_ii.yaml`) optimizes **PR-AUC**
via 5-fold `TimeSeriesSplit` CV on the train split, not ROC-AUC: churn is
imbalanced, and ROC-AUC is optimistic under imbalance in a way that would
misrepresent how the model performs on the minority (churned) class.

**3-way comparison** (held-out **test** snapshot 2011-09-01 — unlike the
Day 4 table above, which reports val-split metrics, these three models are
scored on the test split that CV and tuning never touched, for an
unbiased final comparison):

| Metric | Logistic Regression (baseline) | XGBoost (default) | XGBoost (tuned) |
|--------|---------------------------------|--------------------|-------------------|
| ROC-AUC | 0.777 | 0.746 | 0.784 |
| PR-AUC | 0.810 | 0.758 | 0.799 |
| Precision | 0.775 | 0.723 | 0.721 |
| Recall | 0.680 | 0.726 | 0.844 |
| F1 | 0.725 | 0.724 | 0.778 |
| Brier score | 0.224 | 0.204 | 0.184 |
| Lift @ top 10% | 1.58x | 1.45x | 1.48x |
| Revenue-at-risk (test total) | $85,639 | $164,034 | $239,829 |

Tuning closes most of the gap between XGBoost and the baseline that Day 4
flagged as an open question: PR-AUC improves from 0.758 to 0.799 and Brier
score (calibration) improves from 0.204 to 0.184, but the logistic
regression baseline still edges out tuned XGBoost on PR-AUC (0.810 vs
0.799) on this dataset's ~2,600 training rows. That's an honest result,
not a discrepancy to explain away — with this little training data, more
data or feature engineering is likely a bigger lever than further tuning.

Best hyperparameters found (best CV PR-AUC on train: 0.724):
`n_estimators=400, max_depth=6, learning_rate=0.023, subsample=0.745,
colsample_bytree=0.628, min_child_weight=6, gamma=3.98, reg_alpha=7.53,
reg_lambda=9.89`.

**SHAP explainability** (`shap.TreeExplainer` on the tuned model only —
it's the model that would actually ship, so it's the one worth explaining):
the top 3 features by mean absolute SHAP value are `days_between_txns`,
`recency_days`, and `first_purchase_days` — all measures of transaction
*cadence and tenure*, not raw spend. In business terms, the model's
strongest churn signal is a customer's rhythm going quiet relative to
their own history, not how much they've spent historically: a high-spend
customer who suddenly stops ordering is flagged as high-risk well before
a lower-spend-but-consistent one. That argues for cadence-based retention
triggers ("no order in N days, beyond this customer's usual gap") over
pure spend-tier segmentation.

**Artifacts saved per run** to `outputs/<UTC-timestamp>/`: `model_v1.joblib`
and `metadata.json` (git commit, training date, hyperparameters, split
sizes — full reproducibility), `feature_schema.json` (the contract
`ChurnPredictor.from_artifacts` validates incoming data against),
`best_params.json`, `optuna_study.pkl`, `metrics.json` (all three models),
`feature_importance.csv` (XGBoost-native and mean-|SHAP| side by side), and
`figures/shap_summary.png` / `figures/shap_waterfall_<customer_id>.png`
alongside the tuned model's ROC / PR / calibration plots.

Run the pipeline to reproduce: `python scripts/run_tuning_pipeline.py`
