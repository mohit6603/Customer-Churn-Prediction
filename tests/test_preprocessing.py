"""
Author : Mohit Patle

Description:
Unit tests for src/preprocessing.py — cleaning must drop every
leakage column, repair 'Total Charges' and be safe to reuse on
inference payloads that never had the dropped columns.
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing import clean_raw_data, split_data


def test_leakage_and_junk_columns_dropped(raw_df: pd.DataFrame, config: dict) -> None:
    cleaned = clean_raw_data(raw_df, config)

    for col in ["churn_label", "churn_score", "cltv", "churn_reason",
                "customerid", "count", "country", "state", "city",
                "zip_code", "lat_long", "latitude", "longitude"]:
        assert col not in cleaned.columns

    assert "churn_value" in cleaned.columns  # target survives


def test_total_charges_coerced_and_blank_filled(raw_df: pd.DataFrame, config: dict) -> None:
    cleaned = clean_raw_data(raw_df, config)

    assert pd.api.types.is_numeric_dtype(cleaned["total_charges"])
    # Tenure-0 customer had a blank ' ' -> filled with 0.0, not dropped.
    assert cleaned.loc[0, "total_charges"] == 0.0
    assert len(cleaned) == len(raw_df)


def test_service_no_values_collapsed(raw_df: pd.DataFrame, config: dict) -> None:
    cleaned = clean_raw_data(raw_df, config)

    assert "No phone service" not in set(cleaned["multiple_lines"])
    assert "No internet service" not in set(cleaned["online_security"])


def test_clean_tolerates_inference_payload(raw_df: pd.DataFrame, config: dict) -> None:
    # Inference payloads never contain id/leakage columns.
    payload = raw_df.drop(
        columns=["CustomerID", "Churn Label", "Churn Value", "Churn Score",
                 "CLTV", "Churn Reason"]
    )

    cleaned = clean_raw_data(payload, config)

    assert len(cleaned) == len(payload)
    assert "tenure_months" in cleaned.columns


def test_split_is_stratified(raw_df: pd.DataFrame, config: dict) -> None:
    cleaned = clean_raw_data(raw_df, config)

    X_train, X_test, y_train, y_test = split_data(cleaned, config)

    assert len(X_train) + len(X_test) == len(cleaned)
    assert "churn_value" not in X_train.columns
    # Stratification keeps both classes present in the training split.
    assert set(y_train.unique()) == {0, 1}
