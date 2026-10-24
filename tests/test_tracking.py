"""
Author : Mohit Patle

Description:
Unit tests for src/tracking.py — model versioning, the CSV model
registry and MLflow logging. The project root is redirected to a temp
directory so tests never write into the real models/ or mlruns/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import tracking
from src.tracking import log_mlflow_run, make_model_version, register_model

METRICS: dict[str, Any] = {
    "model": "logistic_regression",
    "trained_at": "2026-07-31T04:07:05+00:00",
    "cv_roc_auc": 0.859,
    "test_at_tuned_threshold": {
        "threshold": 0.29, "roc_auc": 0.849, "recall": 0.917, "precision": 0.434,
    },
}


@pytest.fixture()
def fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tracking, "get_project_root", lambda: tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "reports").mkdir()
    return tmp_path


def test_make_model_version_is_sortable() -> None:
    v1 = make_model_version("2026-07-31T04:07:05+00:00")
    v2 = make_model_version("2026-10-01T00:00:00+00:00")
    assert v1 == "v20260731T040705"
    assert v1 < v2


def test_register_model_versions_and_appends(fake_root: Path, config: dict) -> None:
    latest = fake_root / "models" / "churn_model.joblib"

    latest.write_bytes(b"model-1")
    first = register_model(latest, "v1", METRICS, config)
    latest.write_bytes(b"model-2")
    register_model(latest, "v2", METRICS, config)

    assert first.read_bytes() == b"model-1"  # older versions stay immutable
    registry = pd.read_csv(fake_root / config["artifacts"]["registry_path"])
    assert registry["version"].tolist() == ["v1", "v2"]
    assert registry["model_path"].tolist() == [
        "models/churn_model_v1.joblib", "models/churn_model_v2.joblib",
    ]


def test_mlflow_disabled_is_a_noop(config: dict) -> None:
    disabled = {**config, "tracking": {**config["tracking"], "enabled": False}}
    assert log_mlflow_run(disabled, {}, {}, [], "v1") is None


def test_mlflow_run_is_logged(fake_root: Path, config: dict) -> None:
    mlflow = pytest.importorskip("mlflow")
    report = fake_root / "reports" / "metrics.json"
    report.write_text("{}")

    run_id = log_mlflow_run(
        config, params={"model": "logreg", "C": 2.6},
        metrics={"cv_roc_auc": 0.859}, artifact_paths=[report], run_name="v1",
    )

    run = mlflow.get_run(run_id)
    assert run.data.params == {"model": "logreg", "C": "2.6"}
    assert run.data.metrics == {"cv_roc_auc": 0.859}
    assert run.info.run_name == "v1"
