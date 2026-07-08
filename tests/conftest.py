"""Pytest fixtures."""
import sys
from pathlib import Path

import pytest
import pandas as pd

# dashboard/ isn't an installed package (only src/ is, via the editable
# install) -- add it to sys.path so tests can `import services.*` /
# `import components.*` the same way dashboard/app.py does at runtime.
_DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))


@pytest.fixture
def sample_transactions():
    """Minimal transaction data for testing."""
    return pd.DataFrame(
        {
            "Customer ID": [1, 1, 2, 2, 3],
            "Invoice": ["A", "B", "C", "D", "E"],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-02-01",
                    "2010-01-15",
                    "2010-03-01",
                    "2010-05-01",
                ]
            ),
            "Quantity": [10, 5, 3, 8, 2],
            "Price": [2.5, 3.0, 1.5, 4.0, 2.0],
            "Country": ["UK", "UK", "France", "France", "Germany"],
        }
    )
