"""Shared currency/percentage/date formatting helpers.

Kept as a single flat module (not ``dashboard/utils/``) because ``src/utils``
is already installed as the top-level ``utils`` package (``utils.config``,
``utils.logger``) -- a ``dashboard/utils/`` package would collide with it on
``sys.path`` since Streamlit puts the ``dashboard/`` directory itself on the
path the same way it already does for ``services``/``components``.
"""

from __future__ import annotations

from datetime import date, datetime


def format_currency(value: float | int | None, currency: str = "£") -> str:
    """Format a number as currency, e.g. ``format_currency(1234.5) -> "£1,235"``."""
    if value is None:
        return "N/A"
    return f"{currency}{value:,.0f}"


def format_percent(value: float | int | None, decimals: int = 1) -> str:
    """Format a fraction as a percentage, e.g. ``format_percent(0.123) -> "12.3%"``."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}%}"


def format_date(value: str | date | datetime | None) -> str:
    """Format a date/datetime/ISO string as ``"09 Jul 2026"``; passes through on parse failure."""
    if value is None:
        return "unknown"
    if isinstance(value, (date, datetime)):
        return value.strftime("%d %b %Y")
    try:
        return datetime.fromisoformat(str(value)).strftime("%d %b %Y")
    except ValueError:
        return str(value)


def format_count(value: int | float | None) -> str:
    """Format an integer count with thousands separators."""
    if value is None:
        return "N/A"
    return f"{int(value):,}"
