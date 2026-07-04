"""Pure cleaning functions for the Online Retail II transaction stream.

Every public function takes a :class:`pandas.DataFrame` and returns a fresh
frame with ``reset_index(drop=True)`` so subsequent steps don't carry over
old index gaps. The :func:`clean` orchestrator wires the steps together in
the order documented in the project roadmap (Section 4, Day 2).

Cancellations and returns
-------------------------
The Online Retail II stream records cancellations as negative quantities
(the accompanying "C"-prefixed invoice id is not itself checked here — the
quantity sign is the actual filter condition). We remove these rows at
cleaning time rather than netting them against the original purchase.
Netting would let a late-recorded cancellation silently "undo" a purchase
in our rolling aggregates, which is a classic source of feature leakage in
churn models.
"""
from __future__ import annotations

import pandas as pd

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure cleaners
# ---------------------------------------------------------------------------

def drop_missing_customer_id(
    df: pd.DataFrame, customer_id_column: str
) -> pd.DataFrame:
    """Drop rows whose customer id is missing (NaN / null)."""
    before = len(df)
    cleaned = df.dropna(subset=[customer_id_column])
    _log_step("drop_missing_customer_id", before, len(cleaned))
    return cleaned.reset_index(drop=True)


def remove_negative_quantity(
    df: pd.DataFrame, quantity_column: str
) -> pd.DataFrame:
    """Remove rows with negative quantities (cancellations / returns)."""
    before = len(df)
    cleaned = df[df[quantity_column] >= 0]
    _log_step("remove_negative_quantity", before, len(cleaned))
    return cleaned.reset_index(drop=True)


def remove_zero_price(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """Remove rows with zero or negative price (manual adjustments, errors)."""
    before = len(df)
    cleaned = df[df[price_column] > 0]
    _log_step("remove_zero_price", before, len(cleaned))
    return cleaned.reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    cleaned = df.drop_duplicates()
    _log_step("remove_duplicates", before, len(cleaned))
    return cleaned.reset_index(drop=True)


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "iqr",
    iqr_factor: float = 1.5,
    zscore_threshold: float = 3.0,
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Remove outliers from numeric columns.

    - ``iqr``: keep rows within ``[Q1 - k*IQR, Q3 + k*IQR]`` (NaNs are kept).
    - ``zscore``: keep rows with ``|z| <= threshold`` (NaNs are kept). Uses
      the raw mean/std, so a large contamination fraction can swamp its own
      detector — fine for occasional extreme values, not a robust estimator.
    - ``none``: pass-through.

    ``fit_df``, if given, is used to compute the thresholds instead of
    ``df`` itself, while the resulting mask is still applied to all of
    ``df``. Pass the pre-training-snapshot slice here so outlier bounds
    don't get informed by data the model wouldn't have at training time —
    the same leakage risk as fitting a scaler on the full dataset.
    """
    if method == "none" or not columns:
        return df.reset_index(drop=True)

    stats_source = df if fit_df is None else fit_df
    before = len(df)
    mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            logger.warning("Outlier column not present, skipping: %s", col)
            continue
        if col not in stats_source.columns:
            logger.warning("Outlier fit column not present, skipping: %s", col)
            continue

        fit_values = stats_source[col]
        col_values = df[col]
        if method == "iqr":
            q1 = fit_values.quantile(0.25)
            q3 = fit_values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            mask &= col_values.between(lower, upper) | col_values.isna()
        elif method == "zscore":
            mean = fit_values.mean()
            std = fit_values.std()
            if std == 0 or pd.isna(std):
                continue
            z = (col_values - mean) / std
            mask &= (z.abs() <= zscore_threshold) | col_values.isna()
        else:
            raise ValueError(f"Unknown outlier method: {method!r}")

    cleaned = df[mask]
    _log_step(f"remove_outliers ({method})", before, len(cleaned))
    return cleaned.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Apply the full cleaning pipeline using the flags in ``config.cleaning``.

    Order is intentional:

    1. Drop missing customer ids (so we don't process garbage).
    2. Drop duplicate rows.
    3. Remove negative quantities (cancellations).
    4. Remove zero / negative prices.
    5. Remove numeric outliers.

    Outlier thresholds are fit on transactions up to the training
    snapshot only (``config.features.snapshot_dates["train"]``), then
    applied to every row. Fitting on the full dataset would let the
    distribution of not-yet-available val/test transactions decide what
    counts as an outlier in the training window — the same leakage class
    the feature layer's ``assert_no_future_transactions`` guards against.
    """
    schema = config.dataset_schema
    cleaning = config.cleaning

    cleaned = df.copy()

    if cleaning.drop_missing_customer_id:
        cleaned = drop_missing_customer_id(cleaned, schema.customer_id)
    if cleaning.drop_duplicates:
        cleaned = remove_duplicates(cleaned)
    if cleaning.drop_negative_quantity:
        cleaned = remove_negative_quantity(cleaned, schema.quantity)
    if cleaning.drop_zero_price:
        cleaned = remove_zero_price(cleaned, schema.unit_price)

    if cleaning.outlier_method != "none":
        train_snapshot = pd.Timestamp(config.features.snapshot_dates["train"])
        fit_df = cleaned[cleaned[schema.invoice_date] <= train_snapshot]
        cleaned = remove_outliers(
            cleaned,
            columns=cleaning.outlier_columns,
            method=cleaning.outlier_method,
            fit_df=fit_df,
        )

    logger.info(
        "clean pipeline: %s -> %s rows", f"{len(df):,}", f"{len(cleaned):,}"
    )
    return cleaned


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_step(step: str, before: int, after: int) -> None:
    logger.info("%s: %s -> %s rows (%s dropped)",
                step, f"{before:,}", f"{after:,}", f"{before - after:,}")