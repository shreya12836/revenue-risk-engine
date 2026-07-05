"""Pydantic request/response models mirroring the trained feature schema.

Field-level constraints exist so malformed input (unknown keys, wrong types,
negative counts) fails with a descriptive Pydantic 422 before ever reaching
``ChurnPredictor``, instead of surfacing as an opaque ``ValueError`` from
``_validate_and_order`` or, worse, silently scoring garbage.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from utils.config import load_config

# The 33 feature columns a trained model expects, split by nullability.
# This is the single place these names are listed in source; tests import
# these constants directly instead of retyping the list.
REQUIRED_FEATURE_COLUMNS: list[str] = [
    "frequency",
    "monetary",
    "recency_days",
    "invoice_count_30d",
    "spend_30d",
    "txn_count_30d",
    "invoice_count_60d",
    "spend_60d",
    "txn_count_60d",
    "invoice_count_90d",
    "spend_90d",
    "txn_count_90d",
    "total_invoices",
    "total_txns",
    "total_quantity",
    "total_revenue",
    "tenure_days",
    "first_purchase_days",
    "avg_order_value",
    "avg_basket_size",
    "distinct_products",
]

# Statistical columns that legitimately go NaN for real customers: zero
# transactions in a rolling window, fewer than two points for a std/slope,
# or a single lifetime transaction.
NULLABLE_FEATURE_COLUMNS: list[str] = [
    "avg_spend_30d",
    "avg_basket_30d",
    "spend_std_30d",
    "avg_spend_60d",
    "avg_basket_60d",
    "spend_std_60d",
    "avg_spend_90d",
    "avg_basket_90d",
    "spend_std_90d",
    "spend_slope",
    "txn_count_slope",
    "days_between_txns",
]

FEATURE_COLUMNS: list[str] = REQUIRED_FEATURE_COLUMNS + NULLABLE_FEATURE_COLUMNS

MAX_BATCH_SIZE: int = load_config().api.max_batch_size


class PredictionRequest(BaseModel):
    """One customer's feature vector, matching ``feature_schema.json``.

    ``extra="forbid"`` turns a misspelled or unknown field name into a 422
    instead of the request silently discarding it (and later failing deep
    inside ``ChurnPredictor`` with "missing required feature column").
    """

    model_config = ConfigDict(extra="forbid")

    customer_id: str | int | None = Field(
        default=None,
        description="Optional client-supplied identifier, echoed back as-is. "
        "Not used for any lookup (feature-vector input only, per roadmap MVP scope).",
    )

    # RFM
    frequency: int = Field(ge=0)
    monetary: float = Field(ge=0)
    recency_days: int = Field(ge=0)

    # 30-day rolling window
    invoice_count_30d: float = Field(ge=0)
    spend_30d: float = Field(ge=0)
    avg_spend_30d: float | None = Field(default=None, ge=0)
    spend_std_30d: float | None = Field(default=None, ge=0)
    txn_count_30d: float = Field(ge=0)
    avg_basket_30d: float | None = Field(default=None, ge=0)

    # 60-day rolling window
    invoice_count_60d: float = Field(ge=0)
    spend_60d: float = Field(ge=0)
    avg_spend_60d: float | None = Field(default=None, ge=0)
    spend_std_60d: float | None = Field(default=None, ge=0)
    txn_count_60d: float = Field(ge=0)
    avg_basket_60d: float | None = Field(default=None, ge=0)

    # 90-day rolling window
    invoice_count_90d: float = Field(ge=0)
    spend_90d: float = Field(ge=0)
    avg_spend_90d: float | None = Field(default=None, ge=0)
    spend_std_90d: float | None = Field(default=None, ge=0)
    txn_count_90d: float = Field(ge=0)
    avg_basket_90d: float | None = Field(default=None, ge=0)

    # Lifetime aggregates
    total_invoices: int = Field(ge=0)
    total_txns: int = Field(ge=0)
    total_quantity: int = Field(ge=0)
    total_revenue: float = Field(ge=0)
    tenure_days: int = Field(ge=0)
    first_purchase_days: int = Field(ge=0)
    avg_order_value: float = Field(ge=0)
    avg_basket_size: float = Field(ge=0)
    distinct_products: int = Field(ge=0)

    # Trend / spacing -- slopes can legitimately be negative (declining trend)
    spend_slope: float | None = None
    txn_count_slope: float | None = None
    days_between_txns: float | None = Field(default=None, ge=0)


class BatchPredictionRequest(BaseModel):
    """A bounded list of feature vectors scored in one call."""

    model_config = ConfigDict(extra="forbid")

    records: list[PredictionRequest] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"1 to {MAX_BATCH_SIZE} feature vectors (configurable via "
        "api.max_batch_size).",
    )


class PredictionResponse(BaseModel):
    customer_id: str | int | None = None
    churn_probability: float = Field(ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str
