"""
Author : Mohit Patle

Description:
Feature engineering and the sklearn preprocessing pipeline.

Two kinds of transforms live here, kept strictly separate:

1. ``add_features`` — stateless, row-wise engineered features. Safe to
   apply before the train/test split and reused verbatim at inference.
2. ``build_preprocessor`` — *fitted* transforms (scaling, one-hot
   encoding). These live inside a sklearn ColumnTransformer so they are
   fitted on training folds only, which prevents train/test leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)

# Snake_case names of the add-on service columns (post-cleaning).
ADDON_COLUMNS: list[str] = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

TENURE_BINS: list[float] = [-0.1, 12, 24, 48, np.inf]
TENURE_LABELS: list[str] = ["0-1yr", "1-2yr", "2-4yr", "4yr+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a cleaned dataframe.

    Features
    --------
    num_addon_services : int
        How many of the six add-on services the customer subscribes to.
        Customers embedded in the ecosystem switch providers less.
    avg_monthly_spend : float
        Lifetime average bill (total_charges / tenure_months). For
        tenure-0 customers this falls back to monthly_charges.
    charge_growth : float
        Current bill minus lifetime average. A positive value means the
        bill went up recently — a classic churn trigger.
    tenure_bucket : str
        Tenure grouped into business-friendly bands; captures the
        non-linear "new customers churn most" effect for linear models.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from ``preprocessing.clean_raw_data``.

    Returns
    -------
    pd.DataFrame
        Copy of the input with the engineered columns appended.
    """
    out = df.copy()

    addon_cols = [c for c in ADDON_COLUMNS if c in out.columns]
    out["num_addon_services"] = (
        (out[addon_cols] == "Yes").sum(axis=1).astype(int) if addon_cols else 0
    )

    out["avg_monthly_spend"] = np.where(
        out["tenure_months"] > 0,
        out["total_charges"] / out["tenure_months"].replace(0, 1),
        out["monthly_charges"],
    )
    out["charge_growth"] = out["monthly_charges"] - out["avg_monthly_spend"]

    out["tenure_bucket"] = pd.cut(
        out["tenure_months"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype(str)

    logger.info("Added engineered features: %d total columns", out.shape[1])
    return out


def get_feature_lists(df: pd.DataFrame, target: str = "churn_value") -> tuple[list[str], list[str]]:
    """
    Infer numeric and categorical feature names from dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe (target column allowed but excluded).
    target : str
        Target column name to exclude from both lists.

    Returns
    -------
    tuple[list[str], list[str]]
        (numeric_features, categorical_features)
    """
    features = df.drop(columns=[target], errors="ignore")
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """
    Build the fitted-transform stage of the modelling pipeline.

    Numeric features are standardised (needed by logistic regression;
    harmless for trees). Categorical features are one-hot encoded with
    ``handle_unknown='ignore'`` so an unseen category at inference time
    encodes to all-zeros instead of crashing the service.

    Parameters
    ----------
    numeric_features : list[str]
        Names of numeric columns.
    categorical_features : list[str]
        Names of categorical columns.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor to place at the front of a Pipeline.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )
