"""Tests for dashboard.services.risk_bands's band summary logic."""

from __future__ import annotations

import pandas as pd

from services.risk_bands import band_risk_summary


class TestBandRiskSummary:
    def test_buckets_into_requested_number_of_bands(self):
        customers = pd.DataFrame(
            {
                "spend_90d": list(range(1, 21)),
                "churn_probability": [0.1 * (i % 10) for i in range(20)],
                "revenue_at_risk": [float(i) for i in range(20)],
            }
        )

        summary = band_risk_summary(customers, "spend_90d", n_bins=5)

        assert len(summary) == 5
        assert summary["count"].sum() == 20
        assert {"band", "count", "mean_churn_probability", "revenue_at_risk"} <= set(
            summary.columns
        )

    def test_collapses_bands_when_fewer_unique_values_than_bins(self):
        customers = pd.DataFrame(
            {
                "spend_90d": [10.0] * 10,
                "churn_probability": [0.5] * 10,
                "revenue_at_risk": [5.0] * 10,
            }
        )

        summary = band_risk_summary(customers, "spend_90d", n_bins=5)

        assert not summary.empty
        assert summary["count"].sum() == 10

    def test_empty_dataframe_returns_empty_summary_without_raising(self):
        customers = pd.DataFrame(columns=["spend_90d", "churn_probability", "revenue_at_risk"])

        summary = band_risk_summary(customers, "spend_90d", n_bins=5)

        assert summary.empty
