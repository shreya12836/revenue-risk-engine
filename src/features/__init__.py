"""Time-aware, leakage-safe feature engineering.

The functions in this package compute customer-level features from a frame of
transactions. **Every public feature function enforces the snapshot cutoff
itself** so that callers cannot accidentally leak future information into the
feature matrix. The orchestrator (:func:`features.builder.build_features`)
is the only place that needs to receive both past and future data; feature
functions only ever see the past.

Module layout
-------------

- :mod:`features.rfm`        — recency, frequency, monetary aggregates.
- :mod:`features.rolling`    — windowed aggregations (30/60/90 days).
- :mod:`features.customer_stats` — tenure, AOV, basket size, distinct counts.
- :mod:`features.trend`      — slope / acceleration features over time.
- :mod:`features.labels`     — churn and CLV labels derived from post-snapshot data.
- :mod:`features.builder`    — orchestrator that ties everything together.
- :mod:`features.splits`     — time-based train/val/test split construction.
"""
from __future__ import annotations

from features.builder import build_features, merge_features
from features.customer_stats import calculate_customer_statistics
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

__all__ = [
    "FeatureLabelPair",
    "TimeSplit",
    "build_features",
    "build_labels",
    "build_time_splits",
    "calculate_customer_statistics",
    "calculate_rfm",
    "calculate_rolling_features",
    "calculate_trend_features",
    "merge_features",
    "save_feature_names",
]
