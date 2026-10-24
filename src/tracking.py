"""
Author : Mohit Patle

Description:
Experiment tracking and a lightweight model registry.

1. MLflow tracking — every training run logs its parameters, metrics and
   report artifacts so runs can be compared side by side
   (``mlflow ui``). MLflow is optional: if it is not installed or is
   disabled in the config, training still runs and simply skips it.
2. Model registry — every trained model is saved as an immutable,
   versioned file (models/churn_model_<version>.joblib) and recorded in
   reports/model_registry.csv. models/churn_model.joblib always points
   at the latest version, and rolling back is a single file copy.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger, get_project_root

logger = get_logger(__name__)

REGISTRY_COLUMNS: list[str] = [
    "version", "model", "trained_at", "cv_roc_auc", "test_roc_auc",
    "test_recall", "test_precision", "threshold", "model_path",
]


def make_model_version(trained_at: str) -> str:
    """
    Derive a sortable version id from the training timestamp.

    Example: '2026-07-31T04:07:05+00:00' -> 'v20260731T040705'.
    """
    return "v" + datetime.fromisoformat(trained_at).strftime("%Y%m%dT%H%M%S")


def log_mlflow_run(
    config: dict[str, Any],
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[Path],
    run_name: str,
) -> str | None:
    """
    Log one training run to MLflow.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration with the ``tracking`` section.
    params : dict[str, Any]
        Hyperparameters and run settings.
    metrics : dict[str, float]
        Flat numeric metrics.
    artifact_paths : list[Path]
        Report files (plots, CSVs, JSON) to attach to the run.
    run_name : str
        Display name of the run, usually the model version.

    Returns
    -------
    str | None
        The MLflow run id, or None when tracking is disabled/unavailable.
    """
    tracking_cfg = config.get("tracking", {})
    if not tracking_cfg.get("enabled", False):
        logger.info("MLflow tracking disabled in config; skipping")
        return None
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed; skipping experiment tracking")
        return None

    root = get_project_root()
    mlflow.set_tracking_uri(f"sqlite:///{root / tracking_cfg['db_path']}")
    experiment = mlflow.get_experiment_by_name(tracking_cfg["experiment"])
    experiment_id = (
        experiment.experiment_id if experiment else mlflow.create_experiment(
            tracking_cfg["experiment"],
            artifact_location=(root / tracking_cfg["artifact_dir"]).as_uri(),
        )
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifact_paths:
            if path.exists():
                mlflow.log_artifact(str(path))
    logger.info("Logged MLflow run %s (%s)", run.info.run_id, run_name)
    return run.info.run_id


def register_model(
    artifact_path: Path, version: str, metrics: dict[str, Any], config: dict[str, Any]
) -> Path:
    """
    Store an immutable copy of the model and append it to the registry.

    Parameters
    ----------
    artifact_path : Path
        The freshly written "latest" artifact (models/churn_model.joblib).
    version : str
        Version id from ``make_model_version``.
    metrics : dict[str, Any]
        Metrics summary produced by the training pipeline.
    config : dict[str, Any]
        Project configuration with the ``artifacts`` section.

    Returns
    -------
    Path
        Path of the versioned model file.
    """
    root = get_project_root()
    versioned = artifact_path.with_name(f"{artifact_path.stem}_{version}{artifact_path.suffix}")
    shutil.copy2(artifact_path, versioned)

    tuned = metrics["test_at_tuned_threshold"]
    row = pd.DataFrame([{
        "version": version,
        "model": metrics["model"],
        "trained_at": metrics["trained_at"],
        "cv_roc_auc": round(metrics["cv_roc_auc"], 4),
        "test_roc_auc": round(tuned["roc_auc"], 4),
        "test_recall": round(tuned["recall"], 4),
        "test_precision": round(tuned["precision"], 4),
        "threshold": round(tuned["threshold"], 4),
        "model_path": str(versioned.relative_to(root)),
    }], columns=REGISTRY_COLUMNS)

    registry_path = root / config["artifacts"]["registry_path"]
    if registry_path.exists():
        row = pd.concat([pd.read_csv(registry_path), row], ignore_index=True)
    row.to_csv(registry_path, index=False)

    logger.info("Registered model %s -> %s", version, versioned)
    return versioned
