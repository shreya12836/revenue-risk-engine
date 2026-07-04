"""Project configuration: YAML loading with Pydantic validation.

Each top-level section in the YAML is mapped to a typed Pydantic model.
``load_config`` returns a fully validated :class:`ProjectConfig` instance so
downstream code can rely on attribute access and strict type checking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class DatasetConfig(BaseModel):
    name: str
    source_url: str
    local_path: str
    file_type: Literal["excel", "csv", "parquet"]
    # For Excel sources: ``None`` reads *all* sheets and concatenates them
    # (the common case for multi-period datasets like Online Retail II);
    # a string reads one sheet by name; a list of strings reads those sheets.
    sheet_name: str | list[str] | None = None


class SchemaConfig(BaseModel):
    customer_id: str
    invoice_id: str
    invoice_date: str
    quantity: str
    unit_price: str
    country: str


class CleaningConfig(BaseModel):
    drop_missing_customer_id: bool
    drop_negative_quantity: bool
    drop_zero_price: bool
    drop_duplicates: bool
    outlier_method: Literal["iqr", "zscore", "none"]
    outlier_columns: list[str] = Field(default_factory=list)

    @field_validator("outlier_columns")
    @classmethod
    def _validate_outlier_columns(cls, value: list[str]) -> list[str]:
        # IQR/zscore only make sense on numeric columns. Empty list is allowed
        # when outlier_method == "none".
        if not value:
            return value
        return [c for c in value if c]


class FeaturesConfig(BaseModel):
    snapshot_dates: dict[str, str]
    churn_window_days: int = Field(gt=0)
    clv_window_days: int = Field(gt=0)
    rolling_windows: list[int] = Field(default_factory=list)
    use_log_transform: bool
    scaler: Literal["standard", "minmax", "robust"]

    @field_validator("snapshot_dates")
    @classmethod
    def _validate_snapshot_keys(cls, value: dict[str, str]) -> dict[str, str]:
        required = {"train", "val", "test"}
        missing = required - set(value.keys())
        if missing:
            raise ValueError(
                f"snapshot_dates missing required keys: {sorted(missing)}"
            )
        return value


class HyperparameterTuningConfig(BaseModel):
    n_trials: int = Field(gt=0)
    timeout: int = Field(gt=0)


class ModelingConfig(BaseModel):
    target: Literal["churn", "clv"]
    test_size: float = Field(gt=0.0, lt=1.0)
    cv_folds: int = Field(gt=1)
    use_smote: bool
    # MVP allows only xgboost; lightgbm is deferred to v2 per roadmap.
    models: list[Literal["xgboost"]]
    hyperparameter_tuning: HyperparameterTuningConfig


class OutputConfig(BaseModel):
    model_dir: str
    figures_dir: str
    tables_dir: str


class ProjectConfig(BaseModel):
    # ``protected_namespaces`` silences the harmless Pydantic v2 warning that
    # would otherwise trigger from any model-prefixed attribute. We name the
    # dataset-schema field ``dataset_schema`` (not ``schema``) because the
    # latter would shadow :py:meth:`pydantic.BaseModel.schema` and break any
    # caller that tries to dump JSON Schema from the config.
    model_config = ConfigDict(protected_namespaces=())

    dataset: DatasetConfig
    dataset_schema: SchemaConfig
    cleaning: CleaningConfig
    features: FeaturesConfig
    modeling: ModelingConfig
    output: OutputConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("configs/online_retail_ii.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load and validate the project YAML config into a ``ProjectConfig``."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping: {config_path}"
        )

    # Pydantic raises ``ValidationError`` on any structural problem.
    return ProjectConfig.model_validate(raw, by_name=True)
