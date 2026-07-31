"""
Author : Mohit Patle

Description:
Unit tests for src/inference.py — batch and single prediction against
a tiny pipeline trained on the synthetic fixture, so tests stay fast
and never depend on the serialized production artifact.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import add_features, build_preprocessor, get_feature_lists
from src.inference import _risk_band, predict_batch, predict_single
from src.model_training import build_pipeline
from src.preprocessing import clean_raw_data


@pytest.fixture()
def artifact(raw_df: pd.DataFrame, config: dict) -> dict[str, Any]:
    """A minimal but structurally complete model artifact."""
    df = add_features(clean_raw_data(raw_df, config))
    X, y = df.drop(columns=["churn_value"]), df["churn_value"]

    pipeline = build_pipeline(
        build_preprocessor(*get_feature_lists(X)),
        LogisticRegression(max_iter=500),
    )
    pipeline.fit(X, y)

    return {
        "model": pipeline,
        "model_name": "test_logreg",
        "threshold": 0.5,
        "input_columns": list(X.columns),
        "trained_at": "test",
    }


def test_predict_batch_shapes_and_ranges(
    raw_df: pd.DataFrame, artifact: dict[str, Any]
) -> None:
    scored = predict_batch(raw_df, artifact)

    assert len(scored) == len(raw_df)
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["churn_prediction"].unique()) <= {0, 1}
    assert set(scored["risk_band"].unique()) <= {"Low", "Medium", "High"}
    # Output is ranked for the retention team: highest risk first.
    assert scored["churn_probability"].is_monotonic_decreasing


def test_predict_single_returns_expected_keys(
    raw_df: pd.DataFrame, artifact: dict[str, Any]
) -> None:
    record = raw_df.drop(
        columns=["Churn Label", "Churn Value", "Churn Score", "CLTV", "Churn Reason"]
    ).iloc[0].to_dict()

    result = predict_single(record, artifact)

    assert set(result) == {"churn_probability", "churn_prediction",
                           "risk_band", "threshold"}
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_missing_required_column_raises(
    raw_df: pd.DataFrame, artifact: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        predict_batch(raw_df.drop(columns=["Contract"]), artifact)


@pytest.mark.parametrize(
    ("probability", "band"),
    [(0.05, "Low"), (0.29, "Low"), (0.3, "Medium"), (0.59, "Medium"),
     (0.6, "High"), (0.99, "High")],
)
def test_risk_bands(probability: float, band: str) -> None:
    assert _risk_band(probability) == band
