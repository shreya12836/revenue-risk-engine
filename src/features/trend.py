"""Trend features: how a customer's behaviour has been *changing*.

While RFM/rolling features describe a snapshot of behaviour, trend features
describe its trajectory — is the customer's spend rising or falling? Are
they buying more or less frequently than before?

These features are computed over the customer's full pre-snapshot history.
Time is anchored to each customer's *first* invoice, so x increases
monotonically with calendar time and slope signs are intuitive (positive
slope = revenue trending up).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.invariants import (
    assert_no_future_transactions,
    assert_required_columns,
    revenue_column,
)


def _safe_linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the OLS slope of ``y`` on ``x``, or NaN if undefined.

    A slope is undefined when there's only one point, or when all ``x``
    values are identical (zero variance). Returning NaN — rather than 0 —
    means the feature can be imputed downstream without being silently wrong.
    """
    if len(x) < 2:
        return float("nan")
    x_var = np.nanstd(x)
    if x_var == 0 or np.isnan(x_var):
        return float("nan")
    # ``np.polyfit`` returns coefficients highest-degree-first: [slope, intercept].
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def calculate_trend_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    customer_id_column: str = "customer_id",
    invoice_id_column: str = "invoice_id",
    invoice_date_column: str = "invoice_date",
    quantity_column: str = "quantity",
    unit_price_column: str = "unit_price",
) -> pd.DataFrame:
    """Per-customer trend features over pre-snapshot history.

    Output columns
    --------------
    - ``spend_slope``         — OLS slope of revenue vs. invoice date.
                                Positive = revenue trending up.
    - ``txn_count_slope``     — OLS slope of cumulative invoice count vs.
                                invoice date. Positive = more invoices.
    - ``days_between_txns``   — mean gap between consecutive invoices,
                                always positive.
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

    work["_revenue"] = revenue_column(work)

    # Anchor x to each customer's *first* invoice date, so x increases
    # monotonically with calendar time. Using ``snapshot - invoice_date``
    # here would flip the slope sign — calendar time goes forward as that
    # quantity goes down — and silently produce features whose meaning is
    # the opposite of the docstring.
    first_invoice = work.groupby("customer_id")["invoice_date"].transform("min")
    work["_days_since_first"] = (
        (work["invoice_date"] - first_invoice).dt.total_seconds() / 86400.0
    )

    invoice_level = (
        work.groupby(["customer_id", "invoice_id"], as_index=False)
        .agg(invoice_days=("_days_since_first", "min"), invoice_revenue=("_revenue", "sum"))
    )

    rows: list[dict] = []
    for customer_id, group in invoice_level.groupby("customer_id"):
        group = group.sort_values("invoice_days")
        x = group["invoice_days"].to_numpy()
        revenue = group["invoice_revenue"].to_numpy()

        spend_slope = _safe_linear_slope(x, revenue)

        # Cumulative invoice count over per-customer time. x is now monotonic
        # in calendar time, so a positive slope on cum-vs-x means the
        # customer is buying more often as time passes.
        cum = np.arange(1, len(group) + 1, dtype=float)
        txn_slope = _safe_linear_slope(x, cum)

        if len(group) >= 2:
            gaps = np.diff(group["invoice_days"].to_numpy())
            days_between = float(np.mean(gaps))
        else:
            days_between = float("nan")

        rows.append({
            "customer_id": customer_id,
            "spend_slope": spend_slope,
            "txn_count_slope": txn_slope,
            "days_between_txns": days_between,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["customer_id"] = out["customer_id"].astype("float64")
    return out
