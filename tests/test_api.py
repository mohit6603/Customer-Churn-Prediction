"""
Author : Mohit Patle

Description:
Integration tests for the FastAPI service. The app is built with a tiny
in-memory model and a temporary SQLite log, so the tests never touch
the production artifact or the real prediction log.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import MIN_DRIFT_SAMPLES, create_app
from src.feature_engineering import add_features
from src.monitoring import build_reference_profile
from src.preprocessing import clean_raw_data

CUSTOMER: dict[str, Any] = {
    "customer_id": "TEST-001",
    "gender": "Female", "senior_citizen": "No", "partner": "Yes",
    "dependents": "No", "tenure_months": 2, "phone_service": "Yes",
    "multiple_lines": "No", "internet_service": "Fiber optic",
    "online_security": "No", "online_backup": "No",
    "device_protection": "No", "tech_support": "No",
    "streaming_tv": "Yes", "streaming_movies": "Yes",
    "contract": "Month-to-month", "paperless_billing": "Yes",
    "payment_method": "Electronic check", "monthly_charges": 89.1,
}


@pytest.fixture()
def client(
    artifact: dict[str, Any], raw_df: pd.DataFrame, config: dict, tmp_path: Path
) -> Iterator[TestClient]:
    X = add_features(clean_raw_data(raw_df, config)).drop(columns=["churn_value"])
    app = create_app(
        artifact=artifact,
        log_path=tmp_path / "predictions.db",
        reference_profile=build_reference_profile(X),
    )
    with TestClient(app) as c:
        yield c


def _csv(df: pd.DataFrame) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("customers.csv", df.to_csv(index=False).encode(), "text/csv")}


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True, "model_version": "vtest"}
    assert "X-Process-Time-Ms" in resp.headers


def test_model_info(client: TestClient) -> None:
    body = client.get("/model/info").json()
    assert body["model_name"] == "test_logreg"
    assert body["threshold"] == 0.5
    assert "tenure_months" in body["input_columns"]


def test_predict_single(client: TestClient) -> None:
    resp = client.post("/predict", json=CUSTOMER)
    assert resp.status_code == 200

    body = resp.json()
    assert body["customer_id"] == "TEST-001"
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert body["recommended_action"]
    assert body["model_version"] == "vtest"


def test_predict_rejects_invalid_category(client: TestClient) -> None:
    resp = client.post("/predict", json={**CUSTOMER, "contract": "Month to month"})
    assert resp.status_code == 422


def test_predict_rejects_negative_charges(client: TestClient) -> None:
    resp = client.post("/predict", json={**CUSTOMER, "monthly_charges": -5})
    assert resp.status_code == 422


def test_batch_is_ranked_and_truncated(client: TestClient, raw_df: pd.DataFrame) -> None:
    resp = client.post("/predict/batch?top_n=3", files=_csv(raw_df))
    assert resp.status_code == 200

    body = resp.json()
    assert body["n_customers"] == len(raw_df)
    assert len(body["predictions"]) == 3
    probs = [p["churn_probability"] for p in body["predictions"]]
    assert probs == sorted(probs, reverse=True)


def test_batch_missing_column_is_422(client: TestClient, raw_df: pd.DataFrame) -> None:
    resp = client.post("/predict/batch", files=_csv(raw_df.drop(columns=["Contract"])))
    assert resp.status_code == 422
    assert "Contract" in resp.json()["detail"] or "contract" in resp.json()["detail"]


def test_predictions_are_logged(client: TestClient, raw_df: pd.DataFrame) -> None:
    assert client.get("/monitoring/stats").json() == {"n_predictions": 0}

    client.post("/predict", json=CUSTOMER)
    client.post("/predict/batch", files=_csv(raw_df))

    stats = client.get("/monitoring/stats").json()
    assert stats["n_predictions"] == 1 + len(raw_df)
    assert stats["model_versions"] == {"vtest": 1 + len(raw_df)}


def test_drift_needs_enough_samples(client: TestClient) -> None:
    client.post("/predict", json=CUSTOMER)
    assert client.get("/monitoring/drift").status_code == 409


def test_drift_flags_shifted_traffic(client: TestClient, raw_df: pd.DataFrame) -> None:
    # Production traffic where every customer pays far more than in training.
    n_copies = MIN_DRIFT_SAMPLES // len(raw_df) + 1
    shifted = pd.concat([raw_df] * n_copies, ignore_index=True)
    shifted["Monthly Charges"] = 180.0
    client.post("/predict/batch", files=_csv(shifted))

    body = client.get("/monitoring/drift").json()
    assert body["n_samples"] == len(shifted)
    assert body["drift_detected"]
    assert "monthly_charges" in body["significant_features"]
