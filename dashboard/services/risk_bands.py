"""Quantile-banded risk summaries (spend / recency / frequency / tenure)."""

from __future__ import annotations

import pandas as pd


def band_risk_summary(
    customers_df: pd.DataFrame, column: str, n_bins: int = 5
) -> pd.DataFrame:
    """Bucket ``customers_df[column]`` into up to ``n_bins`` quantile bands.

    Returns one row per band: label, customer count, mean churn probability,
    total revenue at risk. Uses ``duplicates="drop"`` so a column with fewer
    distinct values than ``n_bins`` (or all-identical values) collapses to
    fewer bands instead of raising.
    """
    working = customers_df[[column, "churn_probability", "revenue_at_risk"]].dropna(
        subset=[column]
    )
    if working.empty:
        return pd.DataFrame(columns=["band", "count", "mean_churn_probability", "revenue_at_risk"])

    if working[column].nunique() < 2:
        # Fewer than 2 distinct values -- qcut can't form more than one edge
        # (pandas' exact failure mode for this varies by version), so skip
        # straight to a single band rather than relying on catching it.
        bands = pd.Series(["all"] * len(working), index=working.index)
    else:
        bands = pd.qcut(working[column], q=n_bins, duplicates="drop").astype(str)

    working = working.assign(band=bands)
    summary = (
        working.groupby("band", observed=True)
        .agg(
            count=("churn_probability", "size"),
            mean_churn_probability=("churn_probability", "mean"),
            revenue_at_risk=("revenue_at_risk", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values("band").reset_index(drop=True)
