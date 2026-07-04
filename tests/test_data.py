"""Tests for the data loading and cleaning pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data.cleaner import (
    clean,
    drop_missing_customer_id,
    remove_duplicates,
    remove_negative_quantity,
    remove_outliers,
    remove_zero_price,
)
from data.loader import _read_file, download_if_missing, load
from utils.config import ProjectConfig, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Realistic-ish raw transactions with one issue per row type."""
    return pd.DataFrame(
        {
            "Customer ID": [1, 1, 2, np.nan, 3, 4, 4, 5],
            "Invoice": ["A", "B", "C", "D", "E", "F", "F", "G"],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-02-01",
                    "2010-01-15",
                    "2010-03-01",
                    "2010-05-01",
                    "2010-06-01",
                    "2010-06-01",
                    "2010-07-01",
                ]
            ),
            "Quantity": [10, 5, 3, -2, 0, 8, 8, 1000],
            "Price": [2.5, 3.0, 1.5, 4.0, 0.0, 2.0, 2.0, 5.0],
            "Country": ["UK", "UK", "France", "France", "Germany", "USA", "USA", "UK"],
        }
    )


@pytest.fixture
def config() -> ProjectConfig:
    return load_config()


# ---------------------------------------------------------------------------
# Pure cleaners
# ---------------------------------------------------------------------------

class TestDropMissingCustomerId:
    def test_drops_nan_customer_ids(self, raw_df):
        cleaned = drop_missing_customer_id(raw_df, "Customer ID")
        assert len(cleaned) == len(raw_df) - 1
        assert cleaned["Customer ID"].notna().all()

    def test_resets_index(self, raw_df):
        raw_df_dropped = raw_df.drop(index=0)
        cleaned = drop_missing_customer_id(raw_df_dropped, "Customer ID")
        assert list(cleaned.index) == list(range(len(cleaned)))

    def test_empty_after_drop(self):
        df = pd.DataFrame({"Customer ID": [np.nan, np.nan]})
        assert len(drop_missing_customer_id(df, "Customer ID")) == 0


class TestRemoveNegativeQuantity:
    def test_drops_negative_rows(self, raw_df):
        cleaned = remove_negative_quantity(raw_df, "Quantity")
        assert (cleaned["Quantity"] >= 0).all()
        assert -2 not in cleaned["Quantity"].values

    def test_keeps_zero_quantity(self):
        df = pd.DataFrame({"Quantity": [0, 1, 2]})
        cleaned = remove_negative_quantity(df, "Quantity")
        assert (cleaned["Quantity"] >= 0).all()
        assert len(cleaned) == 3


class TestRemoveZeroPrice:
    def test_drops_zero_and_negative_prices(self):
        df = pd.DataFrame({"Price": [1.0, 0.0, -1.0, 2.0]})
        cleaned = remove_zero_price(df, "Price")
        assert (cleaned["Price"] > 0).all()
        assert len(cleaned) == 2

    def test_all_valid_passes_through(self):
        df = pd.DataFrame({"Price": [0.5, 1.0, 100.0]})
        cleaned = remove_zero_price(df, "Price")
        assert len(cleaned) == 3


class TestRemoveDuplicates:
    def test_drops_exact_duplicates(self, raw_df):
        cleaned = remove_duplicates(raw_df)
        assert len(cleaned) == len(raw_df) - 1

    def test_no_duplicates_passes_through(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert len(remove_duplicates(df)) == 3


class TestRemoveOutliers:
    def test_iqr_removes_tail_outlier(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        cleaned = remove_outliers(df, ["x"], method="iqr", iqr_factor=1.5)
        assert 100 not in cleaned["x"].values

    def test_zscore_removes_tail_outlier(self):
        # Use a tight threshold + larger sample so a single outlier can't inflate
        # the std enough to hide itself (which is the classic failure mode of
        # z-score on small samples).
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000]})
        cleaned = remove_outliers(df, ["x"], method="zscore", zscore_threshold=1.0)
        assert 1000 not in cleaned["x"].values

    def test_none_passes_through(self):
        df = pd.DataFrame({"x": [1, 2, 3, 1000]})
        cleaned = remove_outliers(df, ["x"], method="none")
        assert len(cleaned) == 4

    def test_unknown_method_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown outlier method"):
            remove_outliers(df, ["x"], method="bogus")

    def test_missing_column_warns_and_skips(self, caplog):
        # data.cleaner's logger has propagate=False (see utils/logger.py),
        # so caplog must attach directly to it, not to the root logger.
        df = pd.DataFrame({"x": [1, 2, 3]})
        with caplog.at_level("WARNING", logger="data.cleaner"):
            cleaned = remove_outliers(df, ["does_not_exist"], method="iqr")
        assert len(cleaned) == 3
        assert "does_not_exist" in caplog.text

    def test_fit_df_restricts_threshold_computation_to_subset(self):
        # Rows 5-7 are a cluster of extreme values that would blow out the
        # IQR bounds (swamping) if they were allowed to influence the fit.
        # Fitting on the well-behaved subset alone still catches them.
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 500, 550, 600]})
        fit_df = df.iloc[:5]
        cleaned = remove_outliers(df, ["x"], method="iqr", fit_df=fit_df)
        assert cleaned["x"].tolist() == [1, 2, 3, 4, 5]

    def test_without_fit_df_the_same_cluster_survives(self):
        # Contrast case: fitting on the full contaminated column lets the
        # cluster of extreme values mask itself.
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 500, 550, 600]})
        cleaned = remove_outliers(df, ["x"], method="iqr")
        assert 500 in cleaned["x"].values


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestClean:
    def test_full_pipeline_strips_problem_rows(self, raw_df, config):
        cleaned = clean(raw_df, config)

        assert cleaned["Customer ID"].notna().all()
        assert (cleaned["Quantity"] >= 0).all()
        assert (cleaned["Price"] > 0).all()
        assert len(cleaned) < len(raw_df)

    def test_pipeline_handles_clean_input_unchanged(self, config):
        df = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3],
                "Invoice": ["A", "B", "C"],
                "InvoiceDate": pd.to_datetime(
                    ["2010-01-01", "2010-02-01", "2010-03-01"]
                ),
                "Quantity": [1, 2, 3],
                "Price": [1.0, 2.0, 3.0],
                "Country": ["UK", "UK", "UK"],
            }
        )
        cleaned = clean(df, config)
        assert len(cleaned) == 3

    def test_pipeline_disabling_outliers_keeps_extreme_values(self, config):
        # Disable outlier removal so the synthetic 1000-row quantity survives.
        config.cleaning.outlier_method = "none"
        df = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3],
                "Invoice": ["A", "B", "C"],
                "InvoiceDate": pd.to_datetime(
                    ["2010-01-01", "2010-02-01", "2010-03-01"]
                ),
                "Quantity": [1, 2, 1000],
                "Price": [1.0, 2.0, 3.0],
                "Country": ["UK", "UK", "UK"],
            }
        )
        cleaned = clean(df, config)
        assert 1000 in cleaned["Quantity"].values

    def test_pipeline_does_not_mutate_input(self, raw_df, config):
        original_len = len(raw_df)
        _ = clean(raw_df, config)
        assert len(raw_df) == original_len

    def test_outlier_bounds_are_fit_only_on_pre_train_snapshot_data(self, config):
        # config's train snapshot is 2010-06-01. Rows after it (2011+) carry
        # extreme quantities; if they were allowed to influence the outlier
        # fit, they could swamp their own detector. Bounds must come only
        # from the tame pre-snapshot rows.
        df = pd.DataFrame({
            "Customer ID": [1, 2, 3, 4, 5, 6],
            "Invoice": ["A", "B", "C", "D", "E", "F"],
            "InvoiceDate": pd.to_datetime([
                "2010-01-01", "2010-02-01", "2010-03-01", "2010-04-01",
                "2011-01-01", "2011-01-02",
            ]),
            "Quantity": [1, 2, 3, 4, 5000, 6000],
            "Price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Country": ["UK"] * 6,
        })
        cleaned = clean(df, config)
        assert 5000 not in cleaned["Quantity"].values
        assert 6000 not in cleaned["Quantity"].values


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestDownloadIfMissing:
    def test_skips_when_file_exists(self, tmp_path):
        cached = tmp_path / "cached.csv"
        cached.write_text("a,b\n1,2\n", encoding="utf-8")

        with patch("data.loader.urlretrieve") as mock_fetch:
            result = download_if_missing("https://example.com/x.csv", cached)

        assert result == cached
        mock_fetch.assert_not_called()

    def test_downloads_when_missing(self, tmp_path):
        target = tmp_path / "fresh.csv"

        def fake_retrieve(url: str, path: str) -> None:
            Path(path).write_text("a,b\n1,2\n", encoding="utf-8")

        with patch("data.loader.urlretrieve", side_effect=fake_retrieve):
            result = download_if_missing("https://example.com/x.csv", target)

        assert result == target
        assert target.exists()


class TestReadFile:
    def test_reads_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        df = _read_file(path, "csv", None)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_reads_parquet(self, tmp_path):
        path = tmp_path / "data.parquet"
        pd.DataFrame({"a": [1, 2]}).to_parquet(path)
        df = _read_file(path, "parquet", None)
        assert list(df["a"]) == [1, 2]

    def test_unsupported_type_raises(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file_type"):
            _read_file(path, "docx", None)

    def test_multi_sheet_excel_concatenates_matching_columns(self, tmp_path):
        path = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(
                writer, sheet_name="Year1", index=False
            )
            pd.DataFrame({"a": [3], "b": ["z"]}).to_excel(
                writer, sheet_name="Year2", index=False
            )

        df = _read_file(path, "excel", None)

        assert len(df) == 3
        assert list(df.columns) == ["a", "b"]

    def test_multi_sheet_excel_rejects_mismatched_columns(self, tmp_path):
        path = tmp_path / "mismatched.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame({"a": [1], "b": ["x"]}).to_excel(
                writer, sheet_name="Year1", index=False
            )
            pd.DataFrame({"a": [2], "c": ["y"]}).to_excel(
                writer, sheet_name="Year2", index=False
            )

        with pytest.raises(ValueError, match="differ from sheet"):
            _read_file(path, "excel", None)


class TestLoad:
    def _write_csv(self, tmp_path: Path, content: str) -> Path:
        path = tmp_path / "online_retail_II.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_reads_csv_and_coerces_dtypes(self, tmp_path, config):
        path = self._write_csv(
            tmp_path,
            (
                "Customer ID,Invoice,InvoiceDate,Quantity,Price,Country\n"
                "1,A,2010-01-01,2,5.0,UK\n"
                "2,B,2010-02-01,3,7.5,France\n"
            ),
        )
        config.dataset.file_type = "csv"
        config.dataset.local_path = str(path)

        df = load(config)

        assert len(df) == 2
        assert pd.api.types.is_datetime64_any_dtype(df[config.dataset_schema.invoice_date])
        assert pd.api.types.is_numeric_dtype(df[config.dataset_schema.quantity])
        assert pd.api.types.is_numeric_dtype(df[config.dataset_schema.unit_price])

    def test_load_rejects_missing_required_columns(self, tmp_path, config):
        path = self._write_csv(
            tmp_path,
            "Customer ID,Invoice,Quantity\n1,A,2\n",
        )
        config.dataset.file_type = "csv"
        config.dataset.local_path = str(path)

        with pytest.raises(ValueError, match="missing required columns"):
            load(config)

    def test_load_coerces_bad_dates_to_nat(self, tmp_path, config):
        path = self._write_csv(
            tmp_path,
            (
                "Customer ID,Invoice,InvoiceDate,Quantity,Price,Country\n"
                "1,A,not-a-date,2,5.0,UK\n"
            ),
        )
        config.dataset.file_type = "csv"
        config.dataset.local_path = str(path)

        df = load(config)
        assert df[config.dataset_schema.invoice_date].isna().all()