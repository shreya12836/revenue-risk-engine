"""Lifetime customer statistics as-of a snapshot date.

Unlike rolling features these aggregates are computed over *all* of a
customer's history up to and including the snapshot date. They capture
enduring traits (tenure, total spend, breadth of catalogue engagement) that
change slowly and are useful as anchor features for the model.
"""
from __future__ import annotations

import pandas as pd

from features.invariants import (
    assert_no_future_transactions,
    assert_required_columns,
    revenue_column,
)


def calculate_customer_statistics(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    customer_id_column: str = "customer_id",
    invoice_id_column: str = "invoice_id",
    invoice_date_column: str = "invoice_date",
    quantity_column: str = "quantity",
    unit_price_column: str = "unit_price",
    stock_code_column: str = "StockCode",
) -> pd.DataFrame:
    """Per-customer lifetime statistics.

    Output columns
    --------------
    - ``tenure_days``         — days between first and last invoice.
    - ``first_purchase_days`` — days from first purchase to the snapshot.
    - ``total_invoices``      — count of distinct invoices.
    - ``total_txns``          — count of all line-item transactions.
    - ``total_revenue``       — sum of revenue across the lifetime.
    - ``avg_order_value``     — mean revenue per invoice.
    - ``avg_basket_size``     — mean quantity per transaction.
    - ``distinct_products``   — count of distinct StockCodes purchased.
    - ``total_quantity``      — total units purchased.
    """
    work = df.rename(columns={
        customer_id_column: "customer_id",
        invoice_id_column: "invoice_id",
        invoice_date_column: "invoice_date",
        quantity_column: "quantity",
        unit_price_column: "unit_price",
    }).copy()

    assert_required_columns(work)
    assert_no_future_transactions(work, snapshot_date, "invoice_date")

    stock_col = work[stock_code_column] if stock_code_column in work.columns else None

    work["_revenue"] = revenue_column(work)
    snapshot = pd.Timestamp(snapshot_date)

    grouped = work.groupby("customer_id").agg(
        first_invoice=("invoice_date", "min"),
        last_invoice=("invoice_date", "max"),
        total_invoices=("invoice_id", "nunique"),
        total_txns=("invoice_id", "count"),
        total_quantity=("quantity", "sum"),
        total_revenue=("_revenue", "sum"),
    )
    grouped["tenure_days"] = (
        grouped["last_invoice"] - grouped["first_invoice"]
    ).dt.days
    grouped["first_purchase_days"] = (snapshot - grouped["first_invoice"]).dt.days
    grouped["avg_order_value"] = grouped["total_revenue"] / grouped["total_invoices"]
    grouped["avg_basket_size"] = grouped["total_quantity"] / grouped["total_txns"]

    if stock_col is not None:
        distinct_products = (
            work.assign(_stock=stock_col.values)
            .groupby("customer_id")["_stock"]
            .nunique()
            .rename("distinct_products")
        )
        grouped = grouped.join(distinct_products)
    else:
        grouped["distinct_products"] = pd.NA

    grouped = grouped.drop(columns=["first_invoice", "last_invoice"])
    grouped.index = grouped.index.astype("float64")
    return grouped.reset_index()
