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
4. Add baseline churn model training and evaluation.
5. Save inference artifacts and expose a FastAPI prediction endpoint.
6. Add dashboard views for customer risk and revenue exposure.

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

**Verified on real data** (Online Retail II, both sheets, 1,067,371 rows):

| Split | Snapshot | Customers | Features | Churn rate |
|-------|----------|-----------|----------|------------|
| train | 2010-06-01 | 2,577 | 33 | 51.3% |
| val   | 2010-12-01 | 4,096 | 33 | 67.4% |
| test  | 2011-12-01 | 5,649 | 33 | 90.1% |

Run the smoke test to reproduce: `python scripts/smoke_features.py`
