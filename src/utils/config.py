"""Configuration loading helpers."""
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/online_retail_ii.yaml")
REQUIRED_SECTIONS = ("dataset", "schema", "cleaning", "features", "modeling", "output")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate a YAML project configuration file."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        joined_sections = ", ".join(missing_sections)
        raise ValueError(f"Config file is missing required section(s): {joined_sections}")

    return config
