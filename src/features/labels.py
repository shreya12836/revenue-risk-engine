"""Build supervised-learning labels from post-snapshot transactions.

Labels are *always* computed over the window AFTER the snapshot date. They
are kept in their own module so the feature pipeline can be unit-tested in
isolation from any leakage-sensitive label logic.
"""
from __future__ import annotations

import pandas as pd

from features.invariants import (
    assert_required_columns,
    revenue_column,
)


def build_labels(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    churn_window_days: int,
    clv_window_days: int,
    customer_id_column: str = "customer_id",
    invoice_id_column: str = "invoice_id",
    invoice_date_column: str = "invoice_date",
    quantity_column: str = "quantity",
    unit_price_column: str = "unit_price",
) -> pd.DataFrame:
    """Return per-customer churn and CLV labels.

    Parameters
    ----------
    df : pd.DataFrame
        The *full* transaction frame (pre + post snapshot). This function
        uses data after ``snapshot_date`` and is therefore the only label-
        building step that needs the future window.
    snapshot_date : pd.Timestamp
        The date at which the labels "start". The label window opens strictly
        *after* this date.
    churn_window_days : int
        Number of days into the future over which churn is observed.
    clv_window_days : int
        Number of days into the future over which CLV is summed.

    Returns
    -------
    pd.DataFrame
        Frame indexed by ``customer_id`` with two columns:

        - ``churn`` — 1 if the customer has zero invoices in
          ``(snapshot, snapshot + churn_window_days]``, else 0. Customers who
          never appear in this window are labelled churn=1 (they did not
          purchase). Customers who never appear in the pre-snapshot history
          are dropped.
        - ``clv`` — sum of revenue in ``(snapshot, snapshot + clv_window_days]``.
          Customers with no post-snapshot activity get 0.
    """
    if churn_window_days <= 0:
        raise ValueError(f"churn_window_days must be > 0, got {churn_window_days}")
    if clv_window_days <= 0:
        raise ValueError(f"clv_window_days must be > 0, got {clv_window_days}")

    work = df.rename(columns={
        customer_id_column: "customer_id",
        invoice_id_column: "invoice_id",
        invoice_date_column: "invoice_date",
        quantity_column: "quantity",
        unit_price_column: "unit_price",
    }).copy()

    # Note: no ``assert_no_future_transactions`` here — by design this
    # function consumes the *future* window to build labels. The leakage
    # guarantee is enforced at the feature-function layer instead.
    assert_required_columns(work)
    work["_revenue"] = revenue_column(work)

    snapshot = pd.Timestamp(snapshot_date)
    churn_upper = snapshot + pd.Timedelta(days=churn_window_days)
    clv_upper = snapshot + pd.Timedelta(days=clv_window_days)

    # Universe of customers = those who had any activity pre-snapshot. We
    # restrict to this set because we have no features to score a brand-new
    # customer who first appears after the snapshot.
    pre = work[work["invoice_date"] <= snapshot]
    customers = pre["customer_id"].dropna().unique()

    if len(customers) == 0:
        return pd.DataFrame(columns=["customer_id", "churn", "clv"])

    in_churn_window = work[
        (work["invoice_date"] > snapshot) & (work["invoice_date"] <= churn_upper)
    ]
    in_clv_window = work[
        (work["invoice_date"] > snapshot) & (work["invoice_date"] <= clv_upper)
    ]

    churn_invoice_count = (
        in_churn_window.groupby("customer_id")["invoice_id"].nunique()
    )
    clv_revenue = (
        in_clv_window.groupby("customer_id")["_revenue"].sum().rename("clv")
    )

    out = pd.DataFrame({"customer_id": customers})
    out = out.merge(churn_invoice_count.rename("__churn_invoices"), how="left", on="customer_id")
    out = out.merge(clv_revenue, how="left", on="customer_id")
    out["churn"] = (out["__churn_invoices"].fillna(0) == 0).astype(int)
    out["clv"] = out["clv"].fillna(0.0)
    out = out.drop(columns=["__churn_invoices"])
    out["customer_id"] = out["customer_id"].astype("float64")
    return out.reset_index(drop=True)
