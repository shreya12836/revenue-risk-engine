"""Tests for project configuration loading."""
from pathlib import Path

import pytest

from utils.config import REQUIRED_SECTIONS, load_config


def test_load_default_config_contains_required_sections():
    config = load_config()

    assert set(REQUIRED_SECTIONS).issubset(config)
    assert config["dataset"]["name"] == "online_retail_ii"
    assert config["schema"]["customer_id"] == "Customer ID"
    assert config["modeling"]["target"] == "churn"


def test_load_config_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing_path)


def test_load_config_rejects_missing_required_sections(tmp_path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text("dataset:\n  name: test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required section"):
        load_config(config_path)


def test_load_config_accepts_path_objects():
    config = load_config(Path("configs/online_retail_ii.yaml"))

    assert config["features"]["churn_window_days"] == 90
