"""
Author : Mohit Patle

Description:
Unit tests for src/data_validation.py — the schema contract must
accept a healthy export and reject corrupted ones loudly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_validation import validate_raw_data


def test_valid_data_passes(raw_df: pd.DataFrame) -> None:
    report = validate_raw_data(raw_df)

    assert report["n_rows"] == 8
    assert report["n_cols"] == 33
    assert report["blank_total_charges"] == 1  # the tenure-0 customer
    assert report["warnings"] == []


def test_empty_dataframe_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_raw_data(pd.DataFrame())


def test_duplicate_customer_id_raises(raw_df: pd.DataFrame) -> None:
    corrupted = raw_df.copy()
    corrupted.loc[1, "CustomerID"] = corrupted.loc[0, "CustomerID"]

    with pytest.raises(ValueError, match="duplicated CustomerID"):
        validate_raw_data(corrupted)


def test_invalid_target_value_raises(raw_df: pd.DataFrame) -> None:
    corrupted = raw_df.copy()
    corrupted.loc[0, "Churn Value"] = 2

    with pytest.raises(ValueError, match="outside"):
        validate_raw_data(corrupted)


def test_missing_target_column_raises(raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Target column"):
        validate_raw_data(raw_df.drop(columns=["Churn Value"]))


def test_label_value_mismatch_is_warned(raw_df: pd.DataFrame) -> None:
    corrupted = raw_df.copy()
    corrupted.loc[0, "Churn Label"] = "No"  # row 0 has Churn Value == 1

    report = validate_raw_data(corrupted)

    assert any("disagree" in w for w in report["warnings"])
