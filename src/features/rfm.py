"""Recency / Frequency / Monetary customer aggregates.

RFM is the canonical starting point for churn modelling: it captures how
*recent* a customer's last purchase was, how *frequently* they buy, and how
much *money* they spend. We compute it as-of a fixed ``snapshot_date`` so the
matrix is reproducible across runs and free of time-travel.
"""
from __future__ import annotations

import pandas as pd

from features.invariants import (
    assert_no_future_transactions,
    assert_required_columns,
    revenue_column,
)


def calculate_rfm(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    customer_id_column: str = "customer_id",
    invoice_id_column: str = "invoice_id",
    invoice_date_column: str = "invoice_date",
    quantity_column: str = "quantity",
    unit_price_column: str = "unit_price",
) -> pd.DataFrame:
    """Return per-customer RFM features as-of ``snapshot_date``.

    The returned frame is indexed by ``customer_id`` and has three columns:

    - ``recency_days`` — days between the snapshot and the customer's last
      invoice (lower = more engaged).
    - ``frequency`` — number of distinct invoices up to and including the
      snapshot date.
    - ``monetary`` — total gross revenue across those invoices.

    The function raises :class:`ValueError` if the input contains any
    transaction dated after ``snapshot_date``.
    """
    assert_required_columns(df.rename(columns={
        customer_id_column: "customer_id",
        invoice_id_column: "invoice_id",
        invoice_date_column: "invoice_date",
        quantity_column: "quantity",
        unit_price_column: "unit_price",
    }))
    assert_no_future_transactions(df, snapshot_date, invoice_date_column)

    snapshot = pd.Timestamp(snapshot_date)
    work = df.rename(columns={
        customer_id_column: "customer_id",
        invoice_id_column: "invoice_id",
        invoice_date_column: "invoice_date",
        quantity_column: "quantity",
        unit_price_column: "unit_price",
    })
    # ``copy`` so we don't mutate the caller's frame.
    work = work[["customer_id", "invoice_id", "invoice_date", "quantity", "unit_price"]].copy()

    last_purchase = (
        work.groupby("customer_id")["invoice_date"].max().rename("last_purchase")
    )
    work["_revenue"] = revenue_column(work)

    grouped = work.groupby("customer_id").agg(
        frequency=("invoice_id", "nunique"),
        monetary=("_revenue", "sum"),
    )

    grouped = grouped.join(last_purchase)
    grouped["recency_days"] = (snapshot - grouped["last_purchase"]).dt.days
    grouped = grouped.drop(columns=["last_purchase"])

    # Cast customer_id index to plain int/float where possible for nicer joins.
    grouped.index = grouped.index.rename("customer_id")
    grouped.index = grouped.index.astype("float64")
    return grouped.reset_index()
