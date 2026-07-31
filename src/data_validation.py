"""
Author : Mohit Patle

Description:
Schema and quality checks for the raw IBM Telco churn dataset.
Validation runs BEFORE any cleaning so that silent data drift
(renamed columns, corrupted exports, duplicated customers) fails
loudly instead of producing a quietly wrong model.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# Contract for the 33-column IBM Telco export. If the upstream file
# changes shape, validation should fail before training starts.
EXPECTED_RAW_COLUMNS: list[str] = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Gender", "Senior Citizen",
    "Partner", "Dependents", "Tenure Months", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing",
    "Payment Method", "Monthly Charges", "Total Charges", "Churn Label",
    "Churn Value", "Churn Score", "CLTV", "Churn Reason",
]


def validate_raw_data(df: pd.DataFrame, target: str = "Churn Value") -> dict[str, Any]:
    """
    Validate the raw dataframe against the expected schema.

    Critical problems (empty data, missing target, duplicate customer
    ids, invalid target values) raise ``ValueError``. Non-critical
    observations are collected into the returned report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded from the Excel export.
    target : str
        Name of the target column.

    Returns
    -------
    dict[str, Any]
        Validation report with row/column counts, missing-value summary
        and any non-critical warnings.

    Raises
    ------
    ValueError
        If a critical validation rule is violated.
    """
    if df.empty:
        raise ValueError("Raw dataframe is empty.")

    missing_cols = sorted(set(EXPECTED_RAW_COLUMNS) - set(df.columns))
    if target in missing_cols:
        raise ValueError(f"Target column '{target}' is missing from the data.")

    if df["CustomerID"].duplicated().any():
        n_dup = int(df["CustomerID"].duplicated().sum())
        raise ValueError(f"Found {n_dup} duplicated CustomerID values.")

    invalid_target = set(df[target].unique()) - {0, 1}
    if invalid_target:
        raise ValueError(f"Target contains values outside {{0, 1}}: {invalid_target}")

    warnings: list[str] = []
    if missing_cols:
        warnings.append(f"Missing expected columns: {missing_cols}")

    extra_cols = sorted(set(df.columns) - set(EXPECTED_RAW_COLUMNS))
    if extra_cols:
        warnings.append(f"Unexpected extra columns: {extra_cols}")

    # 'Churn Label' (Yes/No) must agree with 'Churn Value' (1/0).
    if {"Churn Label", "Churn Value"}.issubset(df.columns):
        mismatch = int(
            ((df["Churn Label"] == "Yes").astype(int) != df["Churn Value"]).sum()
        )
        if mismatch:
            warnings.append(f"Churn Label/Value disagree on {mismatch} rows.")

    # 'Total Charges' arrives as text; blanks are expected for brand-new
    # (tenure 0) customers and are handled downstream in preprocessing.
    blank_total_charges = 0
    if "Total Charges" in df.columns:
        blank_total_charges = int(
            pd.to_numeric(df["Total Charges"], errors="coerce").isna().sum()
        )

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].to_dict()

    report: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "churn_rate": float(df[target].mean()),
        "blank_total_charges": blank_total_charges,
        "missing_values": {k: int(v) for k, v in missing_summary.items()},
        "warnings": warnings,
    }

    logger.info("Raw data validation passed: %s", report)
    return report
