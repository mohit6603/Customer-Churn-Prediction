"""
Author : Mohit Patle

Description:
Data cleaning and train/test splitting.

Cleaning is deliberately *stateless* (pure row-wise transforms with no
fitted parameters), so the exact same function is safe to reuse at
inference time without any risk of train/test leakage. Anything that
must be *fitted* (scaling, encoding) lives in the sklearn Pipeline
built in feature_engineering.py instead.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger

logger = get_logger(__name__)

# Values that mean "No" but are phrased relative to another service.
_SERVICE_NO_VALUES = {"No internet service", "No phone service"}


def _to_snake_case(name: str) -> str:
    """Convert a column name like 'Tenure Months' to 'tenure_months'."""
    return name.strip().lower().replace(" ", "_")


def clean_raw_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean the raw Telco dataframe into a modelling-ready table.

    Steps
    -----
    1. Drop identifier, leakage, constant and geography columns.
    2. Coerce 'Total Charges' to numeric; blank values belong to
       tenure-0 customers who have not been billed yet, so 0.0 is the
       semantically correct fill (not an arbitrary imputation).
    3. Collapse 'No internet service' / 'No phone service' to 'No'
       (the information is already carried by the parent service column).
    4. Rename all columns to snake_case.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (training export or inference payload).
    config : dict
        Project configuration with the ``columns`` section.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe. The target column (snake_case
        ``churn_value``) is kept when present.
    """
    cols_cfg = config["columns"]
    drop_cols = [cols_cfg["id"], *cols_cfg["leakage"], *cols_cfg["constant"], *cols_cfg["geo"]]

    # errors='ignore' lets the same cleaner run on inference payloads
    # where leakage/id columns were never present in the first place.
    out = df.drop(columns=drop_cols, errors="ignore").copy()

    if "Total Charges" in out.columns:
        out["Total Charges"] = pd.to_numeric(out["Total Charges"], errors="coerce").fillna(0.0)

    object_cols = out.select_dtypes(exclude="number").columns
    for col in object_cols:
        out[col] = out[col].replace(list(_SERVICE_NO_VALUES), "No")

    out.columns = [_to_snake_case(c) for c in out.columns]

    logger.info("Cleaned data: %d rows, %d columns", *out.shape)
    return out


def split_data(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a cleaned dataframe into stratified train/test sets.

    Stratification preserves the ~26.5% churn rate in both splits so
    that evaluation metrics are not distorted by sampling noise.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing the target column.
    config : dict
        Project configuration with the ``split`` and ``columns`` sections.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test.

    Raises
    ------
    KeyError
        If the target column is missing from ``df``.
    """
    target = _to_snake_case(config["columns"]["target"])
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    X = df.drop(columns=[target])
    y = df[target]

    split_cfg = config["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg.get("stratify", True) else None,
    )

    logger.info(
        "Split data: train=%d rows (churn %.1f%%), test=%d rows (churn %.1f%%)",
        len(X_train), 100 * y_train.mean(), len(X_test), 100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test
