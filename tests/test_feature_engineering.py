"""
Author : Mohit Patle

Description:
Unit tests for src/feature_engineering.py — engineered features must
be numerically correct, especially at the tenure-0 edge case.
"""

from __future__ import annotations

import pandas as pd

from src.feature_engineering import add_features, build_preprocessor, get_feature_lists
from src.preprocessing import clean_raw_data


def _featured(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    return add_features(clean_raw_data(raw_df, config))


def test_num_addon_services_counts_yes_values(raw_df: pd.DataFrame, config: dict) -> None:
    df = _featured(raw_df, config)

    # Customer 0: Online Security=Yes, Online Backup=Yes, rest No -> 2.
    assert df.loc[0, "num_addon_services"] == 2
    assert df["num_addon_services"].between(0, 6).all()


def test_avg_monthly_spend_tenure_zero_fallback(raw_df: pd.DataFrame, config: dict) -> None:
    df = _featured(raw_df, config)

    # Tenure-0 customer falls back to monthly_charges instead of dividing by 0.
    assert df.loc[0, "avg_monthly_spend"] == df.loc[0, "monthly_charges"]
    # Regular customer: total / tenure.
    expected = df.loc[3, "total_charges"] / df.loc[3, "tenure_months"]
    assert abs(df.loc[3, "avg_monthly_spend"] - expected) < 1e-9


def test_tenure_bucket_labels(raw_df: pd.DataFrame, config: dict) -> None:
    df = _featured(raw_df, config)

    assert df.loc[0, "tenure_bucket"] == "0-1yr"    # tenure 0
    assert df.loc[4, "tenure_bucket"] == "1-2yr"    # tenure 24
    assert df.loc[7, "tenure_bucket"] == "4yr+"     # tenure 72


def test_feature_lists_partition_all_columns(raw_df: pd.DataFrame, config: dict) -> None:
    df = _featured(raw_df, config)

    numeric, categorical = get_feature_lists(df)

    assert "churn_value" not in numeric + categorical
    assert set(numeric + categorical) == set(df.columns) - {"churn_value"}
    assert "monthly_charges" in numeric
    assert "contract" in categorical


def test_preprocessor_transforms_without_error(raw_df: pd.DataFrame, config: dict) -> None:
    df = _featured(raw_df, config)
    X = df.drop(columns=["churn_value"])

    preprocessor = build_preprocessor(*get_feature_lists(X))
    Xt = preprocessor.fit_transform(X)

    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] >= X.shape[1]  # one-hot expands the width
