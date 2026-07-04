"""Tests for project configuration loading and Pydantic validation."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from utils.config import ProjectConfig, load_config


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_load_default_config_returns_project_config():
    config = load_config()

    assert isinstance(config, ProjectConfig)
    assert config.dataset.name == "online_retail_ii"
    assert config.dataset_schema.customer_id == "Customer ID"
    assert config.modeling.target == "churn"


def test_load_config_accepts_path_objects():
    config = load_config(Path("configs/online_retail_ii.yaml"))

    assert config.features.churn_window_days == 90


def test_config_models_field_is_xgboost_only():
    """LightGBM is deferred to v2 per roadmap; MVP config must list xgboost only."""
    config = load_config()

    assert config.modeling.models == ["xgboost"]


def test_config_snapshot_dates_have_all_splits():
    config = load_config()

    assert {"train", "val", "test"}.issubset(
        set(config.features.snapshot_dates.keys())
    )


def test_config_validation_rejects_unknown_outlier_method(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
dataset:
  name: test
  source_url: https://example.com/x.csv
  local_path: data/x.csv
  file_type: csv
schema:
  customer_id: Customer ID
  invoice_id: Invoice
  invoice_date: InvoiceDate
  quantity: Quantity
  unit_price: Price
  country: Country
cleaning:
  drop_missing_customer_id: true
  drop_negative_quantity: true
  drop_zero_price: true
  drop_duplicates: true
  outlier_method: bogus
  outlier_columns: []
features:
  snapshot_dates: {train: "2010-01-01", val: "2010-02-01", test: "2010-03-01"}
  churn_window_days: 90
  clv_window_days: 90
  rolling_windows: [30, 60, 90]
  use_log_transform: true
  scaler: standard
modeling:
  target: churn
  test_size: 0.2
  cv_folds: 5
  use_smote: true
  models: [xgboost]
  hyperparameter_tuning: {n_trials: 10, timeout: 60}
output:
  model_dir: models/
  figures_dir: outputs/figures/
  tables_dir: outputs/tables/
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(bad)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

def test_load_config_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing_path)


def test_load_config_rejects_missing_required_sections(tmp_path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text("dataset:\n  name: test\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)