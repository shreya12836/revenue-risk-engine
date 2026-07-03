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
src/api/              FastAPI application package
src/data/             Data loading, cleaning, and feature engineering
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

1. Stabilize scaffold with tests, docs, and lint-clean utilities.
2. Add config-driven data loading and cleaning.
3. Build customer feature generation and labels.
4. Add baseline churn model training and evaluation.
5. Save inference artifacts and expose a FastAPI prediction endpoint.
6. Add dashboard views for customer risk and revenue exposure.
