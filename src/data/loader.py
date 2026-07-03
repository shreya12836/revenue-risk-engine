"""Raw data loading with caching and schema-aware dtype coercion.

The loader is intentionally read-only and side-effect free beyond the
``download_if_missing`` cache. Cleaning lives in :mod:`data.cleaner`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import pandas as pd

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_if_missing(url: str, local_path: str | Path) -> Path:
    """Download the dataset to ``local_path`` if no file exists there yet.

    Returns the resolved :class:`Path` to the local file. Existing files are
    not overwritten — callers can force a redownload by deleting the cache.
    """
    path = Path(local_path)
    if path.exists() and path.stat().st_size > 0:
        logger.info("Using cached dataset at %s", path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, path)
    urlretrieve(url, path)
    return path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _read_file(path: Path, file_type: str, sheet_name: str | None) -> pd.DataFrame:
    if file_type == "excel":
        return pd.read_excel(path, sheet_name=sheet_name)
    if file_type == "csv":
        return pd.read_csv(path)
    if file_type == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file_type: {file_type!r}")


def _required_columns(config: ProjectConfig) -> list[str]:
    schema = config.schema
    return [
        schema.customer_id,
        schema.invoice_id,
        schema.invoice_date,
        schema.quantity,
        schema.unit_price,
        schema.country,
    ]


def load(config: ProjectConfig) -> pd.DataFrame:
    """Download (if needed), read, and coerce dtypes per the configured schema.

    The returned frame keeps the original column names so downstream code can
    reference them through :class:`utils.config.SchemaConfig`.
    """
    dataset = config.dataset
    schema = config.schema

    local_path = download_if_missing(dataset.source_url, dataset.local_path)
    df = _read_file(local_path, dataset.file_type, dataset.sheet_name)

    required = _required_columns(config)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Loaded data is missing required columns: {missing}"
        )

    # Coerce dtypes. ``errors='coerce'`` turns bad rows into NaT/NaN so the
    # cleaner can drop them with a clear log message.
    df[schema.invoice_date] = pd.to_datetime(df[schema.invoice_date], errors="coerce")
    df[schema.customer_id] = pd.to_numeric(df[schema.customer_id], errors="coerce")
    df[schema.quantity] = pd.to_numeric(df[schema.quantity], errors="coerce")
    df[schema.unit_price] = pd.to_numeric(df[schema.unit_price], errors="coerce")

    logger.info(
        "Loaded %s rows x %s columns from %s",
        f"{len(df):,}",
        df.shape[1],
        local_path,
    )
    return df


# ---------------------------------------------------------------------------
# Helpers exposed for tests
# ---------------------------------------------------------------------------

def _rows_with_columns(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Return a small summary dict — handy for smoke assertions in tests."""
    return {c: int(df[c].notna().sum()) for c in columns}