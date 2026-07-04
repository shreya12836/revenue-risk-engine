"""Shared invariants for leakage-safe feature engineering.

A feature that mixes in a transaction dated after ``snapshot_date`` silently
biases the model. Catching this in production is hard — by then the model has
already learned the wrong thing. The assertions here run cheaply on every
call to a feature function so the failure mode is loud and immediate during
development and tests instead.
"""
from __future__ import annotations

import pandas as pd


# Minimal canonical column set every feature function needs after rename.
# ``country`` is intentionally excluded: it is project-relevant metadata but
# not part of the per-customer RFM/rolling/labels surface, so requiring it
# would make the feature layer less reusable.
REQUIRED_COLUMNS = (
    "customer_id",
    "invoice_id",
    "invoice_date",
    "quantity",
    "unit_price",
)


def assert_no_future_transactions(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    date_column: str = "invoice_date",
) -> None:
    """Raise ``ValueError`` if any transaction is strictly after ``snapshot_date``.

    Feature functions call this as the first thing they do. The check makes
    "no leakage" a *property enforced at call time*, not a discipline that
    callers must remember.
    """
    if date_column not in df.columns:
        raise ValueError(
            f"Frame is missing required date column: {date_column!r}"
        )

    snapshot = pd.Timestamp(snapshot_date)
    future_mask = df[date_column] > snapshot
    if future_mask.any():
        latest = df.loc[future_mask, date_column].max()
        raise ValueError(
            "Feature function received transactions after the snapshot date "
            f"(snapshot={snapshot.date()}, latest future tx={latest}). "
            "This would leak future information into the features."
        )


def assert_sufficient_future_window(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    window_days: int,
    date_column: str = "invoice_date",
) -> None:
    """Raise ``ValueError`` if the data doesn't reach ``window_days`` past
    ``snapshot_date``.

    Without this check, a snapshot placed too close to the end of the
    dataset silently mislabels every non-purchasing customer as churned:
    the absence of *data* (right-censoring) gets mistaken for the absence
    of a *purchase*. Call this before computing a label window, not a
    feature window — features are allowed to end at the data boundary,
    labels are not.
    """
    snapshot = pd.Timestamp(snapshot_date)
    required_through = snapshot + pd.Timedelta(days=window_days)
    max_date = df[date_column].max()

    if pd.isna(max_date) or max_date < required_through:
        available = "no data" if pd.isna(max_date) else f"data only through {max_date.date()}"
        raise ValueError(
            "Insufficient future data to build a full label window: "
            f"snapshot={snapshot.date()} + {window_days}d requires data "
            f"through {required_through.date()}, but {available}. This "
            "snapshot would silently mislabel censored customers as churned."
        )


def assert_required_columns(
    df: pd.DataFrame,
    required: tuple[str, ...] = REQUIRED_COLUMNS,
) -> None:
    """Raise ``ValueError`` if the frame is missing any required column."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Transaction frame is missing required columns: {missing}"
        )


def revenue_column(df: pd.DataFrame) -> pd.Series:
    """Return per-row revenue (quantity * unit_price), unscaled."""
    return df["quantity"] * df["unit_price"]
