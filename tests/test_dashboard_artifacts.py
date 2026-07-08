"""Tests for dashboard.services.artifacts's pure loading/assembly logic.

Streamlit's cache decorators (``st.cache_data``/``st.cache_resource``) work
fine called directly in a pytest process -- no running app/session context is
required for the wrapped function to execute and return a value.
"""

from __future__ import annotations

import json

from services.artifacts import (
    build_artifact_summary,
    figure_path,
    load_feature_importance,
    load_metadata,
    load_metrics,
)
from utils.config import load_config


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestLoadMetadata:
    def test_reads_metadata_json(self, tmp_path):
        _write_json(tmp_path / "metadata.json", {"version": "v1", "git_commit": "abc123"})

        metadata = load_metadata(str(tmp_path))

        assert metadata["version"] == "v1"
        assert metadata["git_commit"] == "abc123"


class TestLoadMetrics:
    def test_reads_metrics_json(self, tmp_path):
        payload = {"xgboost_tuned": {"roc_auc": 0.784, "pr_auc": 0.799}}
        _write_json(tmp_path / "metrics.json", payload)

        metrics = load_metrics(str(tmp_path))

        assert metrics["xgboost_tuned"]["roc_auc"] == 0.784


class TestLoadFeatureImportance:
    def test_reads_feature_importance_csv(self, tmp_path):
        csv_path = tmp_path / "feature_importance.csv"
        csv_path.write_text(
            "feature,xgboost_importance,mean_abs_shap\n"
            "recency_days,0.5,0.3\n"
            "monetary,0.2,0.1\n",
            encoding="utf-8",
        )

        df = load_feature_importance(str(tmp_path))

        assert list(df["feature"]) == ["recency_days", "monetary"]
        assert df["mean_abs_shap"].iloc[0] == 0.3


class TestFigurePath:
    def test_returns_path_when_figure_exists(self, tmp_path):
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "roc_curve.png").write_bytes(b"fake-png")

        result = figure_path(str(tmp_path), "roc_curve.png")

        assert result == figures_dir / "roc_curve.png"

    def test_returns_none_when_figure_missing(self, tmp_path):
        result = figure_path(str(tmp_path), "does_not_exist.png")

        assert result is None


class TestBuildArtifactSummary:
    def test_includes_snapshot_date_and_artifact_dir(self, tmp_path):
        _write_json(
            tmp_path / "metadata.json",
            {"version": "v1", "training_date": "2026-07-05T22:49:38", "git_commit": "abc123"},
        )
        config = load_config("configs/online_retail_ii.yaml")

        summary = build_artifact_summary(str(tmp_path), config)

        assert summary["Model version"] == "v1"
        assert summary["Git commit"] == "abc123"
        assert summary["Snapshot date (customer population)"] == config.features.snapshot_dates["test"]
        assert summary["Active artifact directory"] == str(tmp_path)

    def test_git_commit_falls_back_to_unknown_when_missing(self, tmp_path):
        _write_json(tmp_path / "metadata.json", {"version": "v1", "training_date": "2026-07-05"})
        config = load_config("configs/online_retail_ii.yaml")

        summary = build_artifact_summary(str(tmp_path), config)

        assert summary["Git commit"] == "unknown"
