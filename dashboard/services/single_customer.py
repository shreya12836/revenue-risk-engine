"""Single-customer scoring for Page 3 -- mirrors ``api/main.py``'s ``/predict`` handler.

Reuses ``api.schemas.PredictionRequest`` for validation (free, already-tested
``ge=0``/``extra="forbid"`` constraints) rather than re-deriving field checks,
and builds the scoring DataFrame identically to the FastAPI handler so the
dashboard's single-prediction path is provably the same computation, not a
re-implementation.
"""

from __future__ import annotations

import pandas as pd
import shap

from api.schemas import PredictionRequest
from models.predict import ChurnPredictor
from utils.logger import get_logger

logger = get_logger(__name__)


def build_prediction_request(
    form_values: dict[str, float | int | None], customer_id: str | None
) -> PredictionRequest:
    """Validate form input into a ``PredictionRequest``.

    Raises ``pydantic.ValidationError`` on any invalid field -- callers should
    catch this and surface each error message per-field, the same failure
    shape the API's ``handle_validation_error`` produces.
    """
    return PredictionRequest(customer_id=customer_id, **form_values)


def _to_row(request: PredictionRequest) -> pd.DataFrame:
    """Build the single-row scoring DataFrame, identical to ``api/main.py``'s ``/predict``."""
    row = request.model_dump(exclude={"customer_id"})
    return pd.DataFrame([row], dtype=float)


def score_single_customer(predictor: ChurnPredictor, request: PredictionRequest) -> float:
    """Return the churn probability for one customer.

    Can raise ``ValueError`` if the *active* model's ``feature_schema.json``
    has drifted from ``api.schemas``'s static feature-column constants (e.g. a
    retrain changed the feature set) -- ``ChurnPredictor.predict_proba``'s own
    validation raises this. Callers should catch ``ValueError`` around this
    call rather than assuming Pydantic validation alone guarantees success.
    """
    X = _to_row(request)
    proba = predictor.predict_proba(X)
    return float(proba[0])


def explain_single_customer(
    predictor: ChurnPredictor, request: PredictionRequest
) -> shap.Explanation | None:
    """Attempt a SHAP explanation for one customer; return ``None`` on any failure.

    Kept independently failable from ``score_single_customer`` -- a SHAP
    failure (explainer edge case, library version mismatch) must never take
    down the probability result the caller already has.
    """
    X = _to_row(request)
    try:
        _, shap_values = predictor.predict_with_shap(X)
    except Exception:
        logger.exception("SHAP explanation failed for a single-customer prediction")
        return None
    return shap_values
