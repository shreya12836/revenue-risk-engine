# Revenue Risk Engine Roadmap

## Summary

Build a recruiter-facing, production-oriented ML system that predicts customer churn and estimates revenue-at-risk from transaction history.

Current phase: **Day 3 — Feature Engineering complete, Modeling next**

The MVP should prove strong ML judgment first: leakage-safe features, clean tests, baseline comparison, business metrics, saved artifacts, API inference, and a dashboard that reads persisted results.

## MVP Build Plan

### 1. Foundation ✅ done

- Set up a clean project structure, CI, linting, tests, and configuration.
- Use YAML config validated through Pydantic.
- Use structured logging instead of print statements.
- Keep the README professional and portfolio-facing, with no tutorial-style framing.

### 2. Data Layer ✅ done

- Implement a data loader with dtype validation and safe conversion.
- Test required columns, invalid dtypes, missing files, and successful loading.
- Refactor cleaning into small pure functions:
  - `drop_missing_customer`
  - `remove_negative_quantity`
  - `remove_zero_price`
  - `remove_duplicates`
  - `remove_outliers`
- Orchestrate cleaning through `clean()`.
- Document how cancellations and returns are handled.

### 3. Feature Engineering ✅ done

- Build leakage-safe, time-aware feature generation.
- Generate features only from data available before each snapshot date.
- Use time-based train, validation, and test splits only.
- Persist `feature_names.json`.
- Organize feature logic around:
  - `calculate_rfm`
  - `calculate_time_window_features`
  - `calculate_customer_statistics`
  - `build_labels`
  - `merge_features`
  - `build_features`

Implemented in `src/features/` (`rfm.py`, `rolling.py`, `customer_stats.py`,
`trend.py`, `labels.py`, `builder.py`, `splits.py`, `invariants.py`), with
48 passing tests in `tests/test_features.py` including a dedicated
`TestLeakageIsImpossible` check. Verified against the real dataset via
`python scripts/smoke_features.py`.

### 4. Modeling — next up

- Train a simple baseline before any boosted model.
- Train one boosted model for MVP, preferably XGBoost.
- Use SMOTE only inside CV folds if class imbalance requires it.
- Track:
  - ROC-AUC
  - PR-AUC
  - Precision
  - Recall
  - F1
  - Confusion Matrix
  - Lift at top-k
  - Revenue-at-risk captured
- Add calibration checks:
  - Brier score
  - Calibration curve
- Save model, metrics, parameters, plots, and feature metadata.

### 5. Model Code Structure

Keep the MVP model layer simple:

```text
src/models/
+-- __init__.py
+-- train.py
+-- evaluate.py
+-- predict.py
```

- `train.py`: baseline training and XGBoost training.
- `evaluate.py`: metrics, plots, calibration, lift chart, and SHAP outputs.
- `predict.py`: inference wrapper used by the API.

Defer `base.py`, `registry.py`, `xgb.py`, and `lgbm.py` until the project genuinely needs multiple model families or formal model registration.

### 6. Experiment Artifacts

Use timestamped output folders:

```text
outputs/<timestamp>/
```

Store:

- trained model
- metrics
- parameters
- feature names
- plots
- SHAP artifacts where practical

Keep experiment tracking lightweight for MVP. Defer MLflow unless the project needs heavier tracking later.

### 7. Explainability

- Generate SHAP summary plots for the trained model.
- Persist SHAP values as `shap_values.parquet` when practical.
- Explain feature impact in business terms, not only technical terms.

### 8. API

- Build FastAPI endpoints:
  - `/health`
  - `/predict`
  - `/predict/batch`
- Load the model once at startup.
- Use feature-vector inputs for MVP.
- Design the inference wrapper so future `customer_id`-only prediction can be added without rewriting the API.

### 9. Dashboard

- Build a Streamlit dashboard using persisted artifacts.
- Add a **Model Performance** page with:
  - ROC Curve
  - Precision-Recall Curve
  - Confusion Matrix
  - Precision
  - Recall
  - F1
  - Revenue-at-risk view
  - High-risk customer table

### 10. Documentation

Create:

```text
docs/architecture.md
docs/ml_pipeline.md
docs/api.md
docs/model_card.md
```

Documentation should explain:

- problem framing
- data flow
- leakage prevention
- model choice
- metrics
- revenue-at-risk calculation
- limitations
- future work

## Daily GitHub Workflow

Each day:

- Make one coherent milestone.
- Run available tests and lint checks.
- Commit with a professional message.
- Push only stable work to GitHub.
- Avoid noisy commits like `fix`, `wq`, or `changes`.

## MVP Stop Rule

MVP is complete when:

- tests pass
- baseline model is trained
- boosted model is trained
- timestamped artifacts are saved
- API loads the saved model and returns predictions
- dashboard reads saved artifacts
- leakage tests pass
- README contains metrics, demo path, and business framing
- latest stable milestone is pushed to GitHub

## Deferred Until After MVP

- Olist dataset integration
- LightGBM
- full Optuna tuning
- model registry abstraction
- advanced drift monitoring
- batch scoring CLI
- MLflow or heavier tracking
- customer-id-only API requests

## Assumptions

- MVP starts with Online Retail II only.
- Main branch should stay stable.
- "Production-grade" means production-oriented engineering practices, not a deployed enterprise system.
- Business value is framed as churn prediction to targeted retention, reduced CAC, and higher LTV.
- Revenue-at-risk is calculated as churn probability multiplied by predicted customer value.
