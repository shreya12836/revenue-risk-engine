"""Time-aware train/val/test split construction.

The config defines three snapshot dates — ``train``, ``val``, ``test`` —
in :class:`~utils.config.FeaturesConfig.snapshot_dates`. For each we build
features using *only* data up to that snapshot and labels using *only* data
strictly after it. The result is a leak-free split where the test snapshot's
features never see its labels (or vice versa).

Keeping this in the feature package — rather than the modeling package —
reflects the dependency direction: the splitter needs both the feature
builder and the label builder, but neither needs the splitter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from features.builder import build_features
from features.labels import build_labels
from utils.config import ProjectConfig
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FeatureLabelPair:
    """A snapshot-aligned bundle of features and labels.

    Attributes
    ----------
    features : pd.DataFrame
        Per-customer feature matrix computed *as-of* ``snapshot_date``.
    labels : pd.DataFrame
        Per-customer churn and CLV labels derived from transactions
        *after* ``snapshot_date``.
    """

    features: pd.DataFrame
    labels: pd.DataFrame

    @property
    def joined(self) -> pd.DataFrame:
        """Left-joined features+labels, indexed by ``customer_id``.

        Customers in ``features`` but missing from ``labels`` (e.g. they
        only just entered the population right before the snapshot) will
        appear with NaN labels — this is the expected behaviour, not a
        bug, and downstream code should drop or impute them deliberately.
        """
        return self.features.merge(self.labels, on="customer_id", how="left")


@dataclass
class TimeSplit:
    """The three snapshot-aligned splits.

    Each split is a :class:`FeatureLabelPair` keyed by the snapshot date
    it was built at. Downstream modeling code consumes ``train`` to fit,
    ``val`` to tune, and ``test`` for an unbiased final evaluation.
    """

    train: FeatureLabelPair
    val: FeatureLabelPair
    test: FeatureLabelPair

    def __iter__(self):
        # Lets callers do ``for name, pair in split:`` ergonomically.
        yield "train", self.train
        yield "val", self.val
        yield "test", self.test


def _build_pair(
    df: pd.DataFrame,
    config: ProjectConfig,
    snapshot: pd.Timestamp,
) -> FeatureLabelPair:
    """Build features+labels for a single snapshot."""
    features = build_features(df, config, snapshot_date=snapshot)
    labels = build_labels(
        df,
        snapshot_date=snapshot,
        churn_window_days=config.features.churn_window_days,
        clv_window_days=config.features.clv_window_days,
        customer_id_column=config.dataset_schema.customer_id,
        invoice_id_column=config.dataset_schema.invoice_id,
        invoice_date_column=config.dataset_schema.invoice_date,
        quantity_column=config.dataset_schema.quantity,
        unit_price_column=config.dataset_schema.unit_price,
    )
    logger.info(
        "snapshot=%s: %d customers × %d features; %d labeled",
        snapshot.date(),
        len(features),
        len(features.columns) - 1,
        len(labels),
    )
    return FeatureLabelPair(features=features, labels=labels)


def build_time_splits(
    df: pd.DataFrame,
    config: ProjectConfig,
) -> TimeSplit:
    """Build features and labels for each of the three configured snapshots.

    The function reads ``config.features.snapshot_dates`` and constructs
    a :class:`TimeSplit` whose fields are aligned to those dates. The
    caller is responsible for passing an already-cleaned transaction
    frame — this function does not re-clean.
    """
    snapshots = config.features.snapshot_dates
    train = _build_pair(df, config, pd.Timestamp(snapshots["train"]))
    val = _build_pair(df, config, pd.Timestamp(snapshots["val"]))
    test = _build_pair(df, config, pd.Timestamp(snapshots["test"]))
    return TimeSplit(train=train, val=val, test=test)


def save_feature_names(
    features: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Persist the feature column names as a reviewable JSON artifact.

    Saving the column list makes the contract between feature engineering
    and downstream modeling explicit: the model file plus this JSON is
    enough to reconstruct the exact feature matrix that fed it.
    ``customer_id`` is excluded — it is a join key, not a model input.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_columns = [c for c in features.columns if c != "customer_id"]

    payload = {
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "n_rows": len(features),
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info(
        "Saved %d feature names to %s", len(feature_columns), output_path
    )
    return output_path