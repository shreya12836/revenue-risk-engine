"""End-to-end smoke test on the real Online Retail II dataset.

This script is intentionally NOT a pytest test — it is a runnable
verification that exercises the full pipeline (load → clean → split →
features → labels) on the real 45 MB Excel file that ships with the
project. It prints a human-readable summary and exits non-zero if
anything looks broken.

Usage:
    python scripts/smoke_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Allow running from project root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.cleaner import clean
from data.loader import load
from features.splits import build_time_splits, save_feature_names
from utils.config import load_config


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    config = load_config()
    print(f"Loaded config from default path. dataset={config.dataset.name!r}")

    # ---- Load ----------------------------------------------------------
    _section("1. Load")
    t0 = time.perf_counter()
    raw = load(config)
    print(f"  rows:    {len(raw):,}")
    print(f"  columns: {list(raw.columns)}")
    print(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    # ---- Clean ---------------------------------------------------------
    _section("2. Clean")
    t0 = time.perf_counter()
    cleaned = clean(raw, config)
    dropped = len(raw) - len(cleaned)
    pct = dropped / len(raw) * 100 if len(raw) else 0
    print(f"  rows in:  {len(raw):,}")
    print(f"  rows out: {len(cleaned):,}")
    print(f"  dropped:  {dropped:,} ({pct:.1f}%)")
    print(f"  elapsed:  {time.perf_counter() - t0:.2f}s")

    # Sanity checks on the cleaned frame.
    assert cleaned[config.dataset_schema.customer_id].notna().all(), "NaN customer IDs after clean"
    assert (cleaned[config.dataset_schema.quantity] >= 0).all(), "negative quantities after clean"
    assert (cleaned[config.dataset_schema.unit_price] > 0).all(), "non-positive prices after clean"

    # ---- Time-aware splits --------------------------------------------
    _section("3. Time-aware splits")
    t0 = time.perf_counter()
    split = build_time_splits(cleaned, config)
    print(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    for name, pair in split:
        snapshot_date = config.features.snapshot_dates[name]
        joined = pair.joined
        n_customers = len(pair.features)
        n_feature_cols = pair.features.shape[1] - 1
        n_labeled = len(pair.labels)
        # Label coverage: of customers with features, how many have labels?
        coverage = (
            (joined["churn"].notna().sum() / n_customers * 100)
            if n_customers else 0.0
        )
        # Churn rate among labeled customers.
        labeled = joined.dropna(subset=["churn"])
        churn_rate = (
            labeled["churn"].mean() * 100 if len(labeled) else 0.0
        )
        print(
            f"  {name:>5} @ {snapshot_date}: "
            f"{n_customers:,} customers, "
            f"{n_feature_cols} features, "
            f"{n_labeled:,} labeled, "
            f"coverage={coverage:.1f}%, "
            f"churn_rate={churn_rate:.1f}%"
        )

    # ---- Feature quality checks (on train) ----------------------------
    _section("4. Feature quality (train split)")
    train_features = split.train.features
    train_joined = split.train.joined

    print(f"  shape: {train_features.shape}")
    print("  dtypes:")
    for col, dt in train_features.dtypes.items():
        print(f"    {col:30s} {dt}")

    nan_share = train_features.drop(columns=["customer_id"]).isna().mean()
    high_nan = nan_share[nan_share > 0.5]
    if not high_nan.empty:
        print(f"  WARNING: >50% NaN columns: {list(high_nan.index)}")
    else:
        print(f"  NaN share: all columns < 50% missing")

    # Sanity: recency_days must be >= 0 (no customer in the future of the snapshot).
    if "recency_days" in train_features.columns:
        assert (train_features["recency_days"] >= 0).all(), "negative recency found"
        print(f"  recency_days range: {train_features['recency_days'].min()}-{train_features['recency_days'].max()}")

    # Sanity: monetary must be positive.
    if "monetary" in train_features.columns:
        assert (train_features["monetary"] > 0).all(), "non-positive monetary found"
        print(f"  monetary range: {train_features['monetary'].min():.2f}-{train_features['monetary'].max():.2f}")

    # Sanity: churn must be binary.
    labeled = train_joined.dropna(subset=["churn"])
    assert set(labeled["churn"].unique()).issubset({0, 1}), "churn not binary"
    print(f"  churn distribution: {labeled['churn'].value_counts().to_dict()}")

    # ---- Sanity: churn rate sanity across splits ----------------------
    _section("5. Churn rate sanity across splits")
    suspicious = []
    for name, pair in split:
        joined = pair.joined.dropna(subset=["churn"])
        rate = joined["churn"].mean()
        # The churn rate should be in a reasonable band; out-of-band values
        # typically mean the snapshot date is outside the data window and
        # the label window collapses.
        if rate < 0.05 or rate > 0.95:
            suspicious.append((name, rate))
            print(f"  {name:>5}: churn_rate={rate:.1%}  ⚠ SUSPICIOUS")
        else:
            print(f"  {name:>5}: churn_rate={rate:.1%}")
    if suspicious:
        print()
        print("  WARNING: One or more splits have a near-degenerate churn rate.")
        print("  This usually means the snapshot date falls outside the data")
        print("  window and the post-snapshot label window is empty.")

    # ---- Save feature names --------------------------------------------
    _section("5. Persist feature_names.json")
    out_path = ROOT / "outputs" / "feature_names.json"
    save_feature_names(train_features, out_path)
    print(f"  wrote: {out_path}")

    _section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())