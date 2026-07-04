"""Windowed rolling features as-of a snapshot date.

For each configured window length (e.g. ``[30, 60, 90]`` days) we measure how
active a customer was *inside* that window. The convention is "exclusive of
today, inclusive of the window": a 90-day window ending at the snapshot date
covers ``(snapshot - 90 days, snapshot]``.
"""
from __future__ import annotations

import pandas as pd

from features.invariants import (
    assert_no_future_transactions,
    assert_required_columns,
    revenue_column,
)


def calculate_rolling_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    windows: list[int],
    customer_id_column: str = "customer_id",
    invoice_id_column: str = "invoice_id",
    invoice_date_column: str = "invoice_date",
    quantity_column: str = "quantity",
    unit_price_column: str = "unit_price",
) -> pd.DataFrame:
    """Return per-customer window-aggregated features.

    For each ``w`` in ``windows`` the output includes:

    - ``txn_count_<w>d``       — count of transactions in the window
    - ``invoice_count_<w>d``   — count of distinct invoices in the window
    - ``spend_<w>d``           — total revenue in the window
    - ``avg_spend_<w>d``       — mean per-invoice revenue in the window
    - ``avg_basket_<w>d``      — mean per-transaction quantity in the window
    - ``spend_std_<w>d``       — std-dev of per-invoice revenue in the window
    """
    if not windows or not all(w > 0 for w in windows):
        raise ValueError(f"windows must be a non-empty list of positive ints, got {windows!r}")

    # Rename then validate, so callers can pass either the canonical column
    # names or their project-specific variants.
    work = df.rename(columns={
        customer_id_column: "customer_id",
        invoice_id_column: "invoice_id",
        invoice_date_column: "invoice_date",
        quantity_column: "quantity",
        unit_price_column: "unit_price",
    })
    assert_required_columns(work)
    assert_no_future_transactions(work, snapshot_date, "invoice_date")

    work = work[["customer_id", "invoice_id", "invoice_date", "quantity", "unit_price"]].copy()
    work["_revenue"] = revenue_column(work)

    # Anchor of all customers in the input. We reindex every per-window frame
    # against this set so customers with *no* activity in a window survive
    # as NaN rows — the absence of recent activity is itself a strong churn
    # signal and must reach the model rather than be silently dropped.
    all_customers = pd.Index(work["customer_id"].unique(), name="customer_id")

    snapshot = pd.Timestamp(snapshot_date)
    # Pre-compute invoice-level revenue once so std & mean agree.
    invoice_level = (
        work.groupby(["customer_id", "invoice_id"], as_index=False)
        .agg(invoice_date=("invoice_date", "max"), invoice_revenue=("_revenue", "sum"))
    )

    frames: list[pd.DataFrame] = []
    for w in windows:
        lower = snapshot - pd.Timedelta(days=w)
        in_window = invoice_level[
            (invoice_level["invoice_date"] > lower)
            & (invoice_level["invoice_date"] <= snapshot)
        ]

        txn_level_in_window = work[
            (work["invoice_date"] > lower)
            & (work["invoice_date"] <= snapshot)
        ]

        agg = in_window.groupby("customer_id").agg(
            **{
                f"invoice_count_{w}d": ("invoice_id", "nunique"),
                f"spend_{w}d": ("invoice_revenue", "sum"),
                f"avg_spend_{w}d": ("invoice_revenue", "mean"),
                f"spend_std_{w}d": ("invoice_revenue", "std"),
            }
        )
        txn_agg = txn_level_in_window.groupby("customer_id").agg(
            **{
                f"txn_count_{w}d": ("invoice_id", "count"),
                f"avg_basket_{w}d": ("quantity", "mean"),
            }
        )

        per_window = agg.join(txn_agg, how="outer")
        # Reindex to the full customer set so zero-activity customers survive.
        per_window = per_window.reindex(all_customers)
        frames.append(per_window)

    if not frames:
        return pd.DataFrame({"customer_id": all_customers.astype("float64")})

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer")
    combined = combined.reindex(all_customers)
    combined = combined.reset_index()
    combined["customer_id"] = combined["customer_id"].astype("float64")
    return combined
