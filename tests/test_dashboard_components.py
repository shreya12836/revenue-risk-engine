"""Tests for dashboard.components -- pure figure-building logic only.

Streamlit rendering calls (``st.pyplot``, etc.) aren't exercised here; only
the data -> Figure construction, which is what can actually break silently.
"""

from __future__ import annotations

from matplotlib.figure import Figure

from components.confusion_matrix import confusion_matrix_figure


class TestConfusionMatrixFigure:
    def test_returns_a_matplotlib_figure(self):
        counts = {"tn": 1199, "fp": 948, "fn": 454, "tp": 2452}

        fig = confusion_matrix_figure(counts)

        assert isinstance(fig, Figure)

    def test_handles_all_zero_counts_without_raising(self):
        counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

        fig = confusion_matrix_figure(counts)

        assert isinstance(fig, Figure)
