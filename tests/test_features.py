"""Tests for the leakage-safe, snapshot-aware feature layer.

These tests are intentionally framed around three properties:

1. **Correctness** — aggregates match what we expect on hand-built frames.
2. **Leakage prevention** — every public feature function raises loudly if
   it receives a transaction dated *after* the snapshot. This is the
   headline property of the project; the tests below are the only thing
   standing between the model and a silently biased feature matrix.
3. **Time-aware labels** — ``build_labels`` only ever consumes data from
   the *future* window and is the one function allowed to do so.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from features.builder import build_features, merge_features
from features.customer_stats import calculate_customer_statistics
from features.invariants import (
    REQUIRED_COLUMNS,
    assert_no_future_transactions,
    assert_required_columns,
    assert_sufficient_future_window,
    revenue_column,
)
from features.labels import build_labels
from features.rfm import calculate_rfm
from features.rolling import calculate_rolling_features
from features.splits import (
    FeatureLabelPair,
    TimeSplit,
    build_time_splits,
    save_feature_names,
)
from features.trend import calculate_trend_features


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def snapshot() -> pd.Timestamp:
    """A snapshot date well inside the synthetic transaction window."""
    return pd.Timestamp("2010-06-01")


@pytest.fixture
def transactions(snapshot) -> pd.DataFrame:
    """Synthetic pre-snapshot transactions for three customers.

    The dataset is deliberately small and self-consistent so each test can
    predict the expected aggregates by inspection. The transactions all sit
    *before* the snapshot, so feature functions accept them as-is.
    """
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "invoice_id": [
                "A", "A", "B",  # cust 1: 2 invoices, 3 line items
                "C", "D",        # cust 2: 2 invoices, 2 line items
                "E", "F", "G", "G",  # cust 3: 3 invoices, 4 line items
            ],
            "invoice_date": pd.to_datetime([
                "2010-01-01",  # cust 1
                "2010-01-01",
                "2010-04-01",
                "2010-02-15",  # cust 2
                "2010-05-20",
                "2010-03-01",  # cust 3
                "2010-03-15",
                "2010-05-01",
                "2010-05-01",
            ]),
            "quantity":   [1, 2, 3, 4, 5, 2, 1, 3, 1],
            "unit_price": [10.0, 10.0, 5.0, 7.5, 7.5, 4.0, 4.0, 8.0, 8.0],
            "StockCode":  ["X", "X", "Y", "Z", "Z", "X", "Y", "X", "Y"],
        }
    )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

class TestAssertRequiredColumns:
    def test_passes_when_all_present(self, transactions):
        assert_required_columns(transactions)  # does not raise

    def test_raises_when_column_missing(self):
        df = pd.DataFrame({"customer_id": [1], "invoice_id": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            assert_required_columns(df)

    def test_required_columns_match_canonical_names(self):
        # Guard rail: if someone renames a canonical column in invariants.py,
        # the failure should be loud and immediate.
        assert "customer_id" in REQUIRED_COLUMNS
        assert "invoice_date" in REQUIRED_COLUMNS


class TestAssertNoFutureTransactions:
    def test_passes_when_no_future_rows(self, transactions, snapshot):
        assert_no_future_transactions(transactions, snapshot, "invoice_date")

    def test_passes_when_transaction_on_snapshot(self, snapshot):
        df = pd.DataFrame({"invoice_date": pd.to_datetime([snapshot])})
        assert_no_future_transactions(df, snapshot, "invoice_date")

    def test_raises_when_any_future_row(self, snapshot):
        df = pd.DataFrame({
            "invoice_date": pd.to_datetime([
                "2010-05-01",  # past — fine
                "2010-07-01",  # future — must trip
            ]),
        })
        with pytest.raises(ValueError, match="after the snapshot"):
            assert_no_future_transactions(df, snapshot, "invoice_date")

    def test_raises_when_date_column_missing(self, snapshot):
        df = pd.DataFrame({"invoice_id": ["A"]})
        with pytest.raises(ValueError, match="missing required date column"):
            assert_no_future_transactions(df, snapshot, "invoice_date")


class TestAssertSufficientFutureWindow:
    def test_passes_when_window_fully_covered(self, snapshot):
        df = pd.DataFrame({
            "invoice_date": pd.to_datetime([snapshot + pd.Timedelta(days=90)]),
        })
        assert_sufficient_future_window(df, snapshot, 90, "invoice_date")

    def test_raises_when_data_ends_before_window(self, snapshot):
        df = pd.DataFrame({
            "invoice_date": pd.to_datetime([snapshot + pd.Timedelta(days=8)]),
        })
        with pytest.raises(ValueError, match="Insufficient future data"):
            assert_sufficient_future_window(df, snapshot, 90, "invoice_date")

    def test_raises_when_no_data_at_all(self, snapshot):
        df = pd.DataFrame({"invoice_date": pd.to_datetime([snapshot])})
        with pytest.raises(ValueError, match="Insufficient future data"):
            assert_sufficient_future_window(df, snapshot, 90, "invoice_date")


class TestRevenueColumn:
    def test_quantity_times_unit_price(self):
        df = pd.DataFrame({"quantity": [1, 2, 3], "unit_price": [10.0, 5.0, 2.0]})
        result = revenue_column(df).tolist()
        assert result == [10.0, 10.0, 6.0]


# ---------------------------------------------------------------------------
# RFM
# ---------------------------------------------------------------------------

class TestCalculateRfm:
    def test_aggregates_match_hand_calculated_values(self, transactions, snapshot):
        out = calculate_rfm(transactions, snapshot_date=snapshot)

        # One row per customer.
        assert set(out["customer_id"]) == {1.0, 2.0, 3.0}

        # Index by customer for easier assertions.
        rfm = out.set_index("customer_id")

        # Customer 1: 2 distinct invoices, 1+2+3=6 line items, 1*10+2*10+3*5 = 45.
        assert rfm.loc[1.0, "frequency"] == 2
        assert rfm.loc[1.0, "monetary"] == pytest.approx(45.0)

        # Recency is days from snapshot to last invoice.
        # cust 1's last invoice is 2010-04-01 → (Jun 1 - Apr 1) = 61 days.
        assert rfm.loc[1.0, "recency_days"] == 61

        # Customer 2: 2 invoices, 4*7.5 + 5*7.5 = 67.5, last invoice 2010-05-20 → 12 days.
        assert rfm.loc[2.0, "frequency"] == 2
        assert rfm.loc[2.0, "monetary"] == pytest.approx(67.5)
        assert rfm.loc[2.0, "recency_days"] == 12

    def test_customer_id_is_float64(self, transactions, snapshot):
        out = calculate_rfm(transactions, snapshot_date=snapshot)
        assert out["customer_id"].dtype.kind == "f"

    def test_raises_on_future_transactions(self, transactions, snapshot):
        # Add one transaction dated *after* the snapshot.
        leaked = pd.concat([
            transactions,
            pd.DataFrame({
                "customer_id": [1],
                "invoice_id": ["Z"],
                "invoice_date": pd.to_datetime(["2011-01-01"]),
                "quantity": [1],
                "unit_price": [1.0],
                "StockCode": ["Z"],
            }),
        ], ignore_index=True)
        with pytest.raises(ValueError, match="after the snapshot"):
            calculate_rfm(leaked, snapshot_date=snapshot)

    def test_raises_on_missing_required_column(self, snapshot):
        df = pd.DataFrame({"customer_id": [1], "invoice_date": [snapshot]})
        with pytest.raises(ValueError, match="missing required columns"):
            calculate_rfm(df, snapshot_date=snapshot)


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------

class TestCalculateRollingFeatures:
    def test_columns_match_window_spec(self, transactions, snapshot):
        out = calculate_rolling_features(
            transactions, snapshot_date=snapshot, windows=[30, 90]
        )
        # Each window produces 6 columns + the customer_id join key.
        for w in (30, 90):
            assert f"txn_count_{w}d" in out.columns
            assert f"invoice_count_{w}d" in out.columns
            assert f"spend_{w}d" in out.columns
            assert f"avg_spend_{w}d" in out.columns
            assert f"avg_basket_{w}d" in out.columns
            assert f"spend_std_{w}d" in out.columns

    def test_window_excludes_dates_before_lower_bound(self, transactions, snapshot):
        # All transactions for cust 1 fall in (snapshot-90d, snapshot] EXCEPT
        # the Jan 1 ones which are ~151 days before — they must NOT appear in
        # the 90-day window but they MUST appear in the 365-day window.
        out = calculate_rolling_features(
            transactions, snapshot_date=snapshot, windows=[90, 365]
        )
        rfm = out.set_index("customer_id")

        # cust 1's last invoice is 2010-04-01 → 61 days before snapshot.
        # Within 90-day window: only the Apr-1 invoice (1 invoice, $15 spend).
        assert rfm.loc[1.0, "invoice_count_90d"] == 1
        assert rfm.loc[1.0, "spend_90d"] == pytest.approx(15.0)

        # 365-day window sees both Jan-1 and Apr-1 invoices (2 invoices, $45).
        assert rfm.loc[1.0, "invoice_count_365d"] == 2
        assert rfm.loc[1.0, "spend_365d"] == pytest.approx(45.0)

    def test_customers_outside_window_have_nan(self, snapshot):
        # All transactions sit well outside any 30-day window ending at the
        # snapshot. The customers should still appear with NaN aggregates so
        # downstream imputation can fill the gaps.
        df = pd.DataFrame({
            "customer_id": [1, 1],
            "invoice_id": ["A", "B"],
            "invoice_date": pd.to_datetime(["2009-01-01", "2009-02-01"]),
            "quantity": [1, 1],
            "unit_price": [1.0, 1.0],
        })
        out = calculate_rolling_features(df, snapshot_date=snapshot, windows=[30])
        row = out.set_index("customer_id").loc[1.0]
        assert pd.isna(row["invoice_count_30d"])
        assert pd.isna(row["spend_30d"])

    def test_raises_on_empty_window_list(self, transactions, snapshot):
        with pytest.raises(ValueError, match="non-empty list"):
            calculate_rolling_features(transactions, snapshot_date=snapshot, windows=[])

    def test_raises_on_non_positive_window(self, transactions, snapshot):
        with pytest.raises(ValueError, match="positive ints"):
            calculate_rolling_features(transactions, snapshot_date=snapshot, windows=[0])

    def test_raises_on_future_transactions(self, transactions, snapshot):
        leaked = pd.concat([
            transactions,
            pd.DataFrame({
                "customer_id": [1], "invoice_id": ["Z"],
                "invoice_date": pd.to_datetime(["2011-01-01"]),
                "quantity": [1], "unit_price": [1.0],
            }),
        ], ignore_index=True)
        with pytest.raises(ValueError, match="after the snapshot"):
            calculate_rolling_features(leaked, snapshot_date=snapshot, windows=[30])


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

class TestCalculateTrendFeatures:
    def test_columns_present(self, transactions, snapshot):
        out = calculate_trend_features(transactions, snapshot_date=snapshot)
        assert "spend_slope" in out.columns
        assert "txn_count_slope" in out.columns
        assert "days_between_txns" in out.columns

    def test_spend_slope_positive_when_revenue_growing(self, snapshot):
        # Spend grows over time → positive slope.
        df = pd.DataFrame({
            "customer_id": [1, 1, 1],
            "invoice_id": ["A", "B", "C"],
            "invoice_date": pd.to_datetime([
                "2010-04-01", "2010-04-15", "2010-05-01",
            ]),
            "quantity": [1, 1, 1],
            "unit_price": [10.0, 50.0, 100.0],  # 10 → 50 → 100
        })
        out = calculate_trend_features(df, snapshot_date=snapshot)
        slope = out.set_index("customer_id").loc[1.0, "spend_slope"]
        assert slope > 0

    def test_spend_slope_nan_for_single_transaction(self, snapshot):
        df = pd.DataFrame({
            "customer_id": [1],
            "invoice_id": ["A"],
            "invoice_date": pd.to_datetime(["2010-05-01"]),
            "quantity": [1],
            "unit_price": [10.0],
        })
        out = calculate_trend_features(df, snapshot_date=snapshot)
        slope = out.set_index("customer_id").loc[1.0, "spend_slope"]
        assert pd.isna(slope)

    def test_days_between_txns_matches_observed_gaps(self, snapshot):
        # 3 invoices, gaps of 14 and 14 days → mean = 14.
        df = pd.DataFrame({
            "customer_id": [1, 1, 1],
            "invoice_id": ["A", "B", "C"],
            "invoice_date": pd.to_datetime([
                "2010-04-01", "2010-04-15", "2010-04-29",
            ]),
            "quantity": [1, 1, 1],
            "unit_price": [10.0, 10.0, 10.0],
        })
        out = calculate_trend_features(df, snapshot_date=snapshot)
        gap = out.set_index("customer_id").loc[1.0, "days_between_txns"]
        assert gap == pytest.approx(14.0)

    def test_raises_on_future_transactions(self, transactions, snapshot):
        leaked = pd.concat([
            transactions,
            pd.DataFrame({
                "customer_id": [1], "invoice_id": ["Z"],
                "invoice_date": pd.to_datetime(["2011-01-01"]),
                "quantity": [1], "unit_price": [1.0],
            }),
        ], ignore_index=True)
        with pytest.raises(ValueError, match="after the snapshot"):
            calculate_trend_features(leaked, snapshot_date=snapshot)


# ---------------------------------------------------------------------------
# Customer statistics
# ---------------------------------------------------------------------------

class TestCalculateCustomerStatistics:
    def test_tenure_matches_first_to_last(self, transactions, snapshot):
        out = calculate_customer_statistics(transactions, snapshot_date=snapshot)
        rfm = out.set_index("customer_id")

        # cust 1: first 2010-01-01, last 2010-04-01 → 90 days.
        assert rfm.loc[1.0, "tenure_days"] == 90

    def test_first_purchase_days_matches_snapshot_minus_first(self, transactions, snapshot):
        out = calculate_customer_statistics(transactions, snapshot_date=snapshot)
        rfm = out.set_index("customer_id")

        # cust 2: first 2010-02-15, snapshot 2010-06-01 → 106 days.
        assert rfm.loc[2.0, "first_purchase_days"] == 106

    def test_distinct_products_uses_stock_code(self, transactions, snapshot):
        out = calculate_customer_statistics(transactions, snapshot_date=snapshot)
        rfm = out.set_index("customer_id")

        # cust 1 bought StockCodes {X, Y} → 2 distinct.
        # cust 3 bought {X, Y} → 2 distinct.
        assert rfm.loc[1.0, "distinct_products"] == 2
        assert rfm.loc[3.0, "distinct_products"] == 2

    def test_avg_order_value_equals_total_over_invoices(self, transactions, snapshot):
        out = calculate_customer_statistics(transactions, snapshot_date=snapshot)
        rfm = out.set_index("customer_id")

        # cust 3: revenue = 2*4 + 1*4 + 3*8 + 1*8 = 44, 3 invoices → AOV ~ 14.67.
        assert rfm.loc[3.0, "total_revenue"] == pytest.approx(44.0)
        assert rfm.loc[3.0, "total_invoices"] == 3
        assert rfm.loc[3.0, "avg_order_value"] == pytest.approx(44.0 / 3)

    def test_distinct_products_is_na_when_stock_column_missing(self, snapshot):
        df = pd.DataFrame({
            "customer_id": [1], "invoice_id": ["A"],
            "invoice_date": pd.to_datetime(["2010-05-01"]),
            "quantity": [1], "unit_price": [1.0],
        })
        out = calculate_customer_statistics(df, snapshot_date=snapshot)
        assert pd.isna(out.set_index("customer_id").loc[1.0, "distinct_products"])

    def test_raises_on_future_transactions(self, transactions, snapshot):
        leaked = pd.concat([
            transactions,
            pd.DataFrame({
                "customer_id": [1], "invoice_id": ["Z"],
                "invoice_date": pd.to_datetime(["2011-01-01"]),
                "quantity": [1], "unit_price": [1.0],
            }),
        ], ignore_index=True)
        with pytest.raises(ValueError, match="after the snapshot"):
            calculate_customer_statistics(leaked, snapshot_date=snapshot)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestBuildLabels:
    @pytest.fixture
    def post_snapshot_activity(self, snapshot):
        # Customer 1 makes a purchase 30 days after snapshot → NOT churned.
        # Customer 2 makes NO purchase in churn window → churned.
        # Customer 3 makes a purchase 80 days after → still inside 90d window.
        return pd.DataFrame({
            "customer_id": [1, 3, 3],
            "invoice_id": ["L1", "L2", "L3"],
            "invoice_date": pd.to_datetime([
                snapshot + pd.Timedelta(days=30),
                snapshot + pd.Timedelta(days=80),
                snapshot + pd.Timedelta(days=200),  # outside churn window
            ]),
            "quantity": [1, 1, 1],
            "unit_price": [10.0, 20.0, 5.0],
        })

    def test_churn_flag_flips_on_post_snapshot_activity(
        self, transactions, post_snapshot_activity, snapshot
    ):
        df = pd.concat([transactions, post_snapshot_activity], ignore_index=True)
        labels = build_labels(
            df,
            snapshot_date=snapshot,
            churn_window_days=90,
            clv_window_days=90,
        ).set_index("customer_id")

        # cust 1 has 1 invoice in churn window → churn=0
        assert labels.loc[1.0, "churn"] == 0
        # cust 2 has 0 invoices in churn window → churn=1
        assert labels.loc[2.0, "churn"] == 1
        # cust 3 has 1 invoice in churn window → churn=0
        assert labels.loc[3.0, "churn"] == 0

    def test_clv_sums_only_post_snapshot_revenue(
        self, transactions, post_snapshot_activity, snapshot
    ):
        df = pd.concat([transactions, post_snapshot_activity], ignore_index=True)
        labels = build_labels(
            df,
            snapshot_date=snapshot,
            churn_window_days=90,
            clv_window_days=90,
        ).set_index("customer_id")

        # cust 1: $10 in window → clv = 10
        assert labels.loc[1.0, "clv"] == pytest.approx(10.0)
        # cust 2: nothing in window → clv = 0
        assert labels.loc[2.0, "clv"] == 0.0
        # cust 3: $20 in window (the 80-day invoice); the 200-day invoice is
        # outside the 90-day CLV window so it's excluded.
        assert labels.loc[3.0, "clv"] == pytest.approx(20.0)

    def test_raises_when_no_data_covers_the_label_window(self, transactions, snapshot):
        # `transactions` has no rows at all after the snapshot. With no way
        # to tell "genuinely churned" apart from "we simply don't have that
        # far into the future yet" (right-censoring), this must raise
        # rather than silently label everyone churn=1.
        with pytest.raises(ValueError, match="Insufficient future data"):
            build_labels(
                transactions,
                snapshot_date=snapshot,
                churn_window_days=90,
                clv_window_days=90,
            )

    def test_genuinely_churned_customers_flagged_when_window_is_covered(
        self, transactions, snapshot
    ):
        # Extend the data so the full 90-day window actually exists (a
        # customer buys on day 90), while customers 1-3 still have zero
        # purchases in that window — a real churn signal, not censoring.
        future_anchor = pd.DataFrame({
            "customer_id": [4],
            "invoice_id": ["Z1"],
            "invoice_date": pd.to_datetime([snapshot + pd.Timedelta(days=90)]),
            "quantity": [1],
            "unit_price": [1.0],
        })
        df = pd.concat([transactions, future_anchor], ignore_index=True)
        labels = build_labels(
            df,
            snapshot_date=snapshot,
            churn_window_days=90,
            clv_window_days=90,
        ).set_index("customer_id")
        for cust in (1.0, 2.0, 3.0):
            assert labels.loc[cust, "churn"] == 1
            assert labels.loc[cust, "clv"] == 0.0

    def test_drops_customers_with_no_pre_snapshot_history(self, snapshot):
        # Customer 99 appears only AFTER the snapshot.
        df = pd.DataFrame({
            "customer_id": [99],
            "invoice_id": ["L"],
            "invoice_date": pd.to_datetime([snapshot + pd.Timedelta(days=10)]),
            "quantity": [1],
            "unit_price": [5.0],
        })
        labels = build_labels(
            df, snapshot_date=snapshot, churn_window_days=90, clv_window_days=90
        )
        # No features to score them with, so we drop them.
        assert "customer_id" in labels.columns
        assert 99.0 not in labels["customer_id"].values

    def test_raises_on_non_positive_window(self, transactions, snapshot):
        with pytest.raises(ValueError, match="churn_window_days must be > 0"):
            build_labels(
                transactions,
                snapshot_date=snapshot,
                churn_window_days=0,
                clv_window_days=90,
            )
        with pytest.raises(ValueError, match="clv_window_days must be > 0"):
            build_labels(
                transactions,
                snapshot_date=snapshot,
                churn_window_days=90,
                clv_window_days=-1,
            )


# ---------------------------------------------------------------------------
# Builder / merge
# ---------------------------------------------------------------------------

class TestMergeFeatures:
    def test_outer_joins_keep_partial_customers(self):
        a = pd.DataFrame({"customer_id": [1, 2], "x": [10, 20]})
        b = pd.DataFrame({"customer_id": [2, 3], "y": [200, 300]})
        out = merge_features([a, b]).set_index("customer_id")

        # cust 1 only in frame a, cust 3 only in frame b — both survive.
        assert set(out.index) == {1, 2, 3}
        assert out.loc[1, "x"] == 10
        assert pd.isna(out.loc[1, "y"])
        assert pd.isna(out.loc[3, "x"])
        assert out.loc[3, "y"] == 300

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="at least one frame"):
            merge_features([])

    def test_raises_when_join_key_missing(self):
        a = pd.DataFrame({"customer_id": [1], "x": [1]})
        b = pd.DataFrame({"wrong_key": [2], "y": [2]})
        with pytest.raises(ValueError, match="missing the join key"):
            merge_features([a, b])


class TestBuildFeatures:
    def test_combines_all_modules_and_keeps_only_eligible_customers(self, snapshot):
        # cust 4 first appears AFTER snapshot — they must be dropped.
        df = pd.DataFrame({
            "customer_id": [1, 1, 2, 4],
            "invoice_id":   ["A", "B", "C", "Z"],
            "invoice_date": pd.to_datetime([
                "2010-01-01", "2010-04-01", "2010-02-15", "2011-01-01",
            ]),
            "quantity":   [1, 2, 3, 1],
            "unit_price": [10.0, 5.0, 7.5, 1.0],
        })

        class _StubFeatures:
            rolling_windows = [30]

        class _StubConfig:
            dataset_schema = type("S", (), {
                "customer_id": "customer_id",
                "invoice_id": "invoice_id",
                "invoice_date": "invoice_date",
                "quantity": "quantity",
                "unit_price": "unit_price",
            })()
            features = _StubFeatures()

        out = build_features(df, _StubConfig(), snapshot_date=snapshot)
        assert set(out["customer_id"]) == {1.0, 2.0}  # cust 4 dropped

        # Spot-check that RFM was computed: cust 1's recency is 61 days.
        rfm_row = out[out["customer_id"] == 1.0].iloc[0]
        assert rfm_row["recency_days"] == 61

    def test_empty_input_returns_just_the_join_key(self, snapshot):
        df = pd.DataFrame(columns=[
            "customer_id", "invoice_id", "invoice_date", "quantity", "unit_price"
        ])

        class _StubConfig:
            dataset_schema = type("S", (), {
                "customer_id": "customer_id",
                "invoice_id": "invoice_id",
                "invoice_date": "invoice_date",
                "quantity": "quantity",
                "unit_price": "unit_price",
            })()
            features = type("F", (), {"rolling_windows": [30]})()

        out = build_features(df, _StubConfig(), snapshot_date=snapshot)
        assert list(out.columns) == ["customer_id"]


# ---------------------------------------------------------------------------
# Cross-feature leakage guard (integration-level)
# ---------------------------------------------------------------------------

class TestLeakageIsImpossible:
    """A single end-to-end check that a frame with a future transaction
    fails *every* feature function — the headline property of this layer.
    """

    @pytest.fixture
    def frame_with_future_row(self, transactions):
        return pd.concat([
            transactions,
            pd.DataFrame({
                "customer_id": [1], "invoice_id": ["FUTURE"],
                "invoice_date": pd.to_datetime(["2011-01-01"]),
                "quantity": [1], "unit_price": [1.0],
                "StockCode": ["X"],
            }),
        ], ignore_index=True)

    def test_every_feature_function_rejects_future_data(
        self, frame_with_future_row, snapshot
    ):
        """A future-dated row must trip the invariant on the very first
        feature call — no exceptions. ``build_labels`` is intentionally
        excluded because, by design, it consumes the post-snapshot window.
        """
        leaked = frame_with_future_row

        with pytest.raises(ValueError, match="after the snapshot"):
            calculate_rfm(leaked, snapshot_date=snapshot)
        with pytest.raises(ValueError, match="after the snapshot"):
            calculate_rolling_features(leaked, snapshot_date=snapshot, windows=[30])
        with pytest.raises(ValueError, match="after the snapshot"):
            calculate_customer_statistics(leaked, snapshot_date=snapshot)
        with pytest.raises(ValueError, match="after the snapshot"):
            calculate_trend_features(leaked, snapshot_date=snapshot)


# ---------------------------------------------------------------------------
# Time-based splits
# ---------------------------------------------------------------------------

def _make_stub_config(snapshot_dates: dict[str, str]) -> object:
    """Build a minimal config stub the splitter is happy to consume."""
    return type("StubConfig", (), {
        "dataset_schema": type("S", (), {
            "customer_id": "customer_id",
            "invoice_id": "invoice_id",
            "invoice_date": "invoice_date",
            "quantity": "quantity",
            "unit_price": "unit_price",
        })(),
        "features": type("F", (), {
            "rolling_windows": [30, 90],
            "snapshot_dates": snapshot_dates,
            "churn_window_days": 90,
            "clv_window_days": 90,
        })(),
    })()


class TestBuildTimeSplits:
    @pytest.fixture
    def long_history(self):
        """A transaction frame that spans three snapshot dates.

        Customer activity deliberately straddles each snapshot so the
        churn label flips for at least one customer per split. The trailing
        row (cust 3, 2011-09-01) exists purely so the test snapshot
        (2011-06-01) has a full 90-day window to observe — without it,
        every test-split label would be silently right-censored.
        """
        return pd.DataFrame({
            "customer_id": [
                1, 1,        # cust 1 active through Apr 2010
                2, 2, 2,     # cust 2 active through Dec 2010
                3, 3, 3, 3, 3,  # cust 3 active through Sep 2011
            ],
            "invoice_id": ["A1", "A2", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "C5"],
            "invoice_date": pd.to_datetime([
                "2010-01-15", "2010-04-20",  # cust 1
                "2010-03-01", "2010-06-15", "2010-09-30",  # cust 2
                "2010-07-01", "2010-10-01", "2011-01-15", "2011-03-20", "2011-09-01",  # cust 3
            ]),
            "quantity": [1, 2, 1, 3, 2, 1, 1, 2, 1, 1],
            "unit_price": [10.0, 8.0, 12.0, 11.0, 9.0, 5.0, 6.0, 4.0, 7.0, 3.0],
            "StockCode": ["X"] * 10,
        })

    def test_returns_time_split_with_three_pairs(self, long_history):
        config = _make_stub_config({
            "train": "2010-06-01",
            "val": "2010-12-01",
            "test": "2011-06-01",
        })
        split = build_time_splits(long_history, config)
        assert isinstance(split, TimeSplit)
        for name, pair in split:
            assert isinstance(pair, FeatureLabelPair)
            assert "customer_id" in pair.features.columns

    def test_split_iterates_in_train_val_test_order(self, long_history):
        config = _make_stub_config({
            "train": "2010-06-01",
            "val": "2010-12-01",
            "test": "2011-06-01",
        })
        split = build_time_splits(long_history, config)
        order = [name for name, _ in split]
        assert order == ["train", "val", "test"]

    def test_train_features_only_use_pre_snapshot_data(self, long_history):
        # For train snapshot 2010-06-01: cust 1's last invoice is 2010-04-20,
        # so recency_days = 42. No future data should leak in.
        config = _make_stub_config({
            "train": "2010-06-01",
            "val": "2010-12-01",
            "test": "2011-06-01",
        })
        train = build_time_splits(long_history, config).train
        cust1 = train.features.set_index("customer_id").loc[1.0]
        assert cust1["recency_days"] == 42

    def test_labels_reflect_post_snapshot_activity(self, long_history):
        # For train snapshot 2010-06-01: cust 1 has no post-snapshot activity
        # in the 90-day window (next invoice is 2010-04-20, which is before
        # the snapshot). So cust 1 → churn=1.
        # cust 2 has a post-snapshot invoice on 2010-06-15 (qty 3 × $11 =
        # $33), inside the 90-day window → churn=0, clv=33.
        config = _make_stub_config({
            "train": "2010-06-01",
            "val": "2010-12-01",
            "test": "2011-06-01",
        })
        train = build_time_splits(long_history, config).train
        labels = train.labels.set_index("customer_id")
        assert labels.loc[1.0, "churn"] == 1
        assert labels.loc[2.0, "churn"] == 0
        assert labels.loc[2.0, "clv"] == pytest.approx(33.0)

    def test_joined_contains_both_features_and_labels(self, long_history):
        config = _make_stub_config({
            "train": "2010-06-01",
            "val": "2010-12-01",
            "test": "2011-06-01",
        })
        train = build_time_splits(long_history, config).train
        joined = train.joined
        # Must have both feature columns (recency_days) and label columns.
        assert "recency_days" in joined.columns
        assert "churn" in joined.columns
        assert "clv" in joined.columns


# ---------------------------------------------------------------------------
# Feature-name persistence
# ---------------------------------------------------------------------------

class TestSaveFeatureNames:
    @pytest.fixture
    def feature_frame(self):
        return pd.DataFrame({
            "customer_id": [1.0, 2.0, 3.0],
            "recency_days": [10, 20, 30],
            "frequency": [1, 2, 3],
            "monetary": [100.0, 200.0, 300.0],
        })

    def test_writes_json_with_feature_columns_excluding_customer_id(
        self, feature_frame, tmp_path
    ):
        path = tmp_path / "feature_names.json"
        save_feature_names(feature_frame, path)

        assert path.exists()
        with path.open() as fh:
            payload = json.load(fh)
        assert payload["feature_columns"] == ["recency_days", "frequency", "monetary"]
        assert payload["n_features"] == 3
        assert payload["n_rows"] == 3

    def test_creates_parent_directories(self, feature_frame, tmp_path):
        path = tmp_path / "nested" / "deeper" / "feature_names.json"
        save_feature_names(feature_frame, path)
        assert path.exists()

    def test_returns_written_path(self, feature_frame, tmp_path):
        path = tmp_path / "fn.json"
        result = save_feature_names(feature_frame, path)
        assert result == path
