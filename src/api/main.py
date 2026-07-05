"""Production FastAPI inference service for the tuned churn model.

Wraps ChurnPredictor -- the single inference entry point the training
pipeline, SHAP step, and this API all share -- behind three endpoints so
scoring logic is never duplicated. The model loads once at startup via the
lifespan hook so a broken/missing artifacts directory fails fast at boot
instead of on the first request.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import get_predictor
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from models.predict import ChurnPredictor
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_predictor()  # eager load: fail fast at boot, not on first request
    yield
    get_predictor.cache_clear()


app = FastAPI(
    title="Revenue Risk Engine -- Churn Inference API",
    description="Serves churn-probability predictions from the tuned XGBoost model.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log method/path/status/latency for every request. No payload bodies."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Log only field locations, never the offending raw values -- the
    # response body below still returns FastAPI's normal error shape to
    # the caller, but server-side logs stay free of business-data payloads.
    locations = [error["loc"] for error in exc.errors()]
    logger.warning(
        "Validation failed for %s %s: %d error(s) at %s",
        request.method,
        request.url.path,
        len(exc.errors()),
        locations,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    # Defense-in-depth: ChurnPredictor._validate_and_order can still raise
    # ValueError even after Pydantic validation (e.g. a future schema change).
    logger.warning("Rejected request to %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled error while serving %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.get("/health", response_model=HealthResponse)
async def health(predictor: ChurnPredictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(model_version=predictor.schema["version"])


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest, predictor: ChurnPredictor = Depends(get_predictor)
) -> PredictionResponse:
    row = payload.model_dump(exclude={"customer_id"})
    # dtype=float forces a real NaN for a None-valued nullable column -- a
    # single-row frame otherwise infers `object` dtype (holding literal
    # None) whenever a column's only value is None, which XGBoost rejects.
    X = pd.DataFrame([row], dtype=float)
    probability = float(predictor.predict_proba(X)[0])
    return PredictionResponse(
        customer_id=payload.customer_id, churn_probability=probability
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictionRequest, predictor: ChurnPredictor = Depends(get_predictor)
) -> BatchPredictionResponse:
    rows = [record.model_dump(exclude={"customer_id"}) for record in payload.records]
    X = pd.DataFrame(rows, dtype=float)
    probabilities = predictor.predict_proba(X)
    predictions = [
        PredictionResponse(
            customer_id=record.customer_id, churn_probability=float(prob)
        )
        for record, prob in zip(payload.records, probabilities)
    ]
    return BatchPredictionResponse(predictions=predictions)
