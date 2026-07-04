"""Leakage-safe missing-value imputation.

Fit the imputer on the training split only, then apply the *same* fitted
imputer to validation/test data. Fitting a fresh imputer per split would
leak each split's own distribution into what should be an unbiased
comparison — the same class of bug already fixed in
``data.cleaner.remove_outliers`` (which accepts a ``fit_df`` parameter for
exactly this reason).

Used only by the logistic-regression baseline in ``models.train``; XGBoost
handles missing values natively and is trained on raw features.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer


def fit_imputer(X_train: pd.DataFrame, strategy: str = "median") -> SimpleImputer:
    """Fit a ``SimpleImputer`` on the training features only."""
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_train)
    return imputer


def apply_imputer(imputer: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform ``X`` with an already-fitted imputer, preserving shape and labels."""
    transformed = imputer.transform(X)
    return pd.DataFrame(transformed, columns=X.columns, index=X.index)
