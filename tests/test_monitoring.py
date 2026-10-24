"""
Author : Mohit Patle

Description:
Unit tests for src/monitoring.py — PSI maths, drift detection on
deliberately shifted data, and the SQLite prediction log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_features
from src.inference import predict_batch
from src.monitoring import (
    PredictionLogger,
    build_reference_profile,
    compute_psi,
    detect_drift,
    summarize_drift,
)
from src.preprocessing import clean_raw_data


@pytest.fixture()
def reference() -> pd.DataFrame:
    """A larger synthetic training set with known distributions."""
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame({
        "monthly_charges": rng.normal(65, 20, n),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n,
                               p=[0.55, 0.25, 0.20]),
    })


def test_psi_is_zero_for_identical_distributions() -> None:
    dist = np.array([0.2, 0.3, 0.5])
    assert compute_psi(dist, dist) == pytest.approx(0.0)


def test_psi_grows_with_shift() -> None:
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    small = compute_psi(expected, np.array([0.3, 0.25, 0.25, 0.2]))
    large = compute_psi(expected, np.array([0.7, 0.1, 0.1, 0.1]))
    assert 0 < small < large


def test_psi_handles_empty_bins() -> None:
    assert np.isfinite(compute_psi(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.5, 0.5])))


def test_no_drift_on_same_distribution(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference.iloc[:1000])
    report = detect_drift(profile, reference.iloc[1000:])
    assert (report["status"] == "stable").all()
    assert not summarize_drift(report)["drift_detected"]


def test_drift_detected_on_shifted_data(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference)
    shifted = reference.assign(
        monthly_charges=reference["monthly_charges"] + 40,  # price increase
        contract="Month-to-month",                         # mix collapses
    )
    summary = summarize_drift(detect_drift(profile, shifted))

    assert summary["drift_detected"]
    assert summary["retraining_recommended"]
    assert set(summary["significant_features"]) == {"monthly_charges", "contract"}


def test_unseen_category_counts_as_drift(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference)
    report = detect_drift(profile, reference.assign(contract="Three year"))
    assert report.set_index("feature").at["contract", "status"] == "significant"


def test_profile_covers_all_model_features(raw_df: pd.DataFrame, config: dict) -> None:
    X = add_features(clean_raw_data(raw_df, config)).drop(columns=["churn_value"])
    profile = build_reference_profile(X)

    assert set(profile) == set(X.columns)
    for spec in profile.values():
        total = sum(spec["proportions"].values()) if spec["type"] == "categorical" \
            else sum(spec["proportions"])
        assert total == pytest.approx(1.0)


def test_prediction_logger_roundtrip(
    raw_df: pd.DataFrame, artifact: dict[str, Any], tmp_path: Path
) -> None:
    log = PredictionLogger(tmp_path / "predictions.db")
    log.log(raw_df, predict_batch(raw_df, artifact), model_version="vtest")

    logged = log.load()
    assert len(logged) == len(raw_df)
    assert set(logged["model_version"]) == {"vtest"}
    assert set(log.load_inputs()["CustomerID"]) == set(raw_df["CustomerID"])

    stats = log.stats()
    assert stats["n_predictions"] == len(raw_df)
    assert 0 <= stats["flagged_rate"] <= 1


def test_empty_log_stats(tmp_path: Path) -> None:
    assert PredictionLogger(tmp_path / "p.db").stats() == {"n_predictions": 0}
