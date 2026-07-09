"""Tests for dashboard.formatting's currency/percent/date/count helpers."""

from __future__ import annotations

from formatting import format_count, format_currency, format_date, format_percent


class TestFormatCurrency:
    def test_formats_positive_value(self):
        assert format_currency(1234.6) == "£1,235"

    def test_formats_zero(self):
        assert format_currency(0) == "£0"

    def test_formats_negative_value(self):
        assert format_currency(-50) == "£-50"

    def test_returns_na_for_none(self):
        assert format_currency(None) == "N/A"


class TestFormatPercent:
    def test_formats_fraction(self):
        assert format_percent(0.1234) == "12.3%"

    def test_formats_zero(self):
        assert format_percent(0.0) == "0.0%"

    def test_returns_na_for_none(self):
        assert format_percent(None) == "N/A"


class TestFormatDate:
    def test_formats_iso_string(self):
        assert format_date("2026-07-05T22:49:38") == "05 Jul 2026"

    def test_passes_through_unparseable_string(self):
        assert format_date("not-a-date") == "not-a-date"

    def test_returns_unknown_for_none(self):
        assert format_date(None) == "unknown"


class TestFormatCount:
    def test_formats_with_thousands_separator(self):
        assert format_count(12345) == "12,345"

    def test_returns_na_for_none(self):
        assert format_count(None) == "N/A"
