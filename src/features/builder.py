"""Orchestrator that assembles the per-customer feature matrix.

The orchestrator applies the snapshot-date cutoff once, then dispatches to
each individual feature function. Keeping the cutoff in one place is the
"single point of truth" pattern: every downstream function operates on
already-filtered past data and only has to assert the invariant, not
re-derive it.
"""
from __future__ import annotations

import pandas as pd

from features.customer_stats import calculate_customer_statistics
from features.rfm import calculate_rfm
from features.rolling import calculate_rolling_features
from features.trend import calculate_trend_features
from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def merge_features(frames: list[pd.DataFrame], on: str = "customer_id") -> pd.DataFrame:
    """Outer-join a list of per-customer feature frames on ``on``.

    Each frame may cover a *different* set of customers (e.g. some customers
    never appear in a 30-day window). The outer join keeps everyone and lets
    downstream imputation fill the gaps, which is preferable to silently
    dropping partially-observed customers.
    """
    if not frames:
        raise ValueError("merge_features requires at least one frame")

    merged = frames[0].copy()
    for i, frame in enumerate(frames[1:], start=2):
        if on not in frame.columns:
            raise ValueError(
                f"Frame #{i} is missing the join key {on!r}: {list(frame.columns)}"
            )
        if on in merged.columns:
            merged = merged.merge(frame, on=on, how="outer")
        else:
            merged = merged.merge(frame, on=on, how="outer")
    return merged


def build_features(
    df: pd.DataFrame,
    config: ProjectConfig,
    snapshot_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Compute the full feature matrix as-of ``snapshot_date``.

    The function:

    1. Applies the snapshot-date cutoff (``invoice_date <= snapshot_date``).
    2. Computes RFM, rolling-window, customer-statistics, and trend features.
    3. Outer-joins them on ``customer_id``.

    Customers whose first purchase is *after* ``snapshot_date`` are dropped:
    they have no pre-snapshot history to characterize them, and trying to
    score them would require different missing-value handling best deferred
    to the modeling step.

    The input frame is expected to use the schema's column names (e.g.
    ``Customer ID`` for Online Retail II). Each per-feature function takes
    the schema names as arguments and renames internally, so no upfront
    rename is needed here. If a schema column is missing, the slicing on
    ``schema.invoice_date`` raises ``KeyError`` loudly.
    """
    snapshot = pd.Timestamp(snapshot_date)
    schema = config.dataset_schema
    features_cfg = config.features

    past = df[df[schema.invoice_date] <= snapshot].copy()

    if past.empty:
        logger.warning(
            "build_features: no transactions found at or before %s", snapshot.date()
        )
        return pd.DataFrame(columns=["customer_id"])

    # Drop customers who first appear *after* the snapshot — they have no
    # pre-snapshot history, so no features can be computed honestly.
    first_purchase = past.groupby(schema.customer_id)[schema.invoice_date].min()
    eligible_customers = first_purchase[first_purchase <= snapshot].index
    past = past[past[schema.customer_id].isin(eligible_customers)]

    rfm = calculate_rfm(
        past,
        snapshot_date=snapshot,
        customer_id_column=schema.customer_id,
        invoice_id_column=schema.invoice_id,
        invoice_date_column=schema.invoice_date,
        quantity_column=schema.quantity,
        unit_price_column=schema.unit_price,
    )
    rolling = calculate_rolling_features(
        past,
        snapshot_date=snapshot,
        windows=features_cfg.rolling_windows,
        customer_id_column=schema.customer_id,
        invoice_id_column=schema.invoice_id,
        invoice_date_column=schema.invoice_date,
        quantity_column=schema.quantity,
        unit_price_column=schema.unit_price,
    )
    stats = calculate_customer_statistics(
        past,
        snapshot_date=snapshot,
        customer_id_column=schema.customer_id,
        invoice_id_column=schema.invoice_id,
        invoice_date_column=schema.invoice_date,
        quantity_column=schema.quantity,
        unit_price_column=schema.unit_price,
    )
    trend = calculate_trend_features(
        past,
        snapshot_date=snapshot,
        customer_id_column=schema.customer_id,
        invoice_id_column=schema.invoice_id,
        invoice_date_column=schema.invoice_date,
        quantity_column=schema.quantity,
        unit_price_column=schema.unit_price,
    )

    merged = merge_features([rfm, rolling, stats, trend])

    logger.info(
        "build_features: %d customers × %d features at snapshot %s",
        len(merged),
        len(merged.columns) - 1,
        snapshot.date(),
    )
    return merged
