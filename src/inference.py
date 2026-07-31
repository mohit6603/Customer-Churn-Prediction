"""
Author : Mohit Patle

Description:
Inference utilities: load the serialized model artifact and score new
customers. The Streamlit app and the tests both go through this module,
so training-time cleaning/feature-engineering is guaranteed to be
replayed identically at prediction time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.feature_engineering import add_features
from src.preprocessing import clean_raw_data
from src.utils import get_logger, get_project_root, load_config

logger = get_logger(__name__)

# Probability bands used to translate a score into a business action.
RISK_BANDS: list[tuple[float, str]] = [(0.6, "High"), (0.3, "Medium"), (0.0, "Low")]


def load_artifact(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the serialized model artifact.

    Parameters
    ----------
    path : str | Path | None
        Path to the .joblib artifact. Defaults to the path in
        configs/config.yaml.

    Returns
    -------
    dict[str, Any]
        Artifact with keys: model, threshold, input_columns,
        model_name, metrics, trained_at.

    Raises
    ------
    FileNotFoundError
        If no artifact exists at the resolved path (train first).
    """
    if path is None:
        config = load_config()
        path = get_project_root() / config["artifacts"]["model_path"]
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No model artifact at {path}. Run 'python -m src.train_pipeline' first."
        )

    artifact = joblib.load(path)
    logger.info("Loaded %s artifact trained at %s",
                artifact.get("model_name"), artifact.get("trained_at"))
    return artifact


def prepare_features(df_raw: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Replay the training-time cleaning + feature engineering on new data.

    Parameters
    ----------
    df_raw : pd.DataFrame
        New customer rows using the raw export schema (leakage columns
        may be absent — they are dropped with errors='ignore').
    config : dict | None
        Project configuration; loaded from disk when omitted.

    Returns
    -------
    pd.DataFrame
        Feature dataframe ready for the pipeline.
    """
    config = config or load_config()
    cleaned = clean_raw_data(df_raw, config)
    featured = add_features(cleaned)
    return featured.drop(columns=["churn_value"], errors="ignore")


def predict_batch(
    df_raw: pd.DataFrame, artifact: dict[str, Any] | None = None
) -> pd.DataFrame:
    """
    Score a batch of customers.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Customer rows in the raw export schema.
    artifact : dict[str, Any] | None
        Loaded model artifact; loaded from disk when omitted.

    Returns
    -------
    pd.DataFrame
        Copy of the input with churn_probability, churn_prediction and
        risk_band columns appended, sorted by probability descending.

    Raises
    ------
    ValueError
        If required input columns are missing after preparation.
    """
    artifact = artifact or load_artifact()
    X = prepare_features(df_raw)

    missing = set(artifact["input_columns"]) - set(X.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    X = X[artifact["input_columns"]]

    proba = artifact["model"].predict_proba(X)[:, 1]
    threshold = artifact["threshold"]

    result = df_raw.copy()
    result["churn_probability"] = proba
    result["churn_prediction"] = (proba >= threshold).astype(int)
    result["risk_band"] = [_risk_band(p) for p in proba]

    logger.info("Scored %d customers (%d flagged as churn risk)",
                len(result), int(result["churn_prediction"].sum()))
    return result.sort_values("churn_probability", ascending=False)


def predict_single(
    record: dict[str, Any], artifact: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Score a single customer described as a dict of raw-schema fields.

    Parameters
    ----------
    record : dict[str, Any]
        One customer, e.g. {"Gender": "Female", "Tenure Months": 3, ...}.
    artifact : dict[str, Any] | None
        Loaded model artifact; loaded from disk when omitted.

    Returns
    -------
    dict[str, Any]
        churn_probability, churn_prediction, risk_band and the
        decision threshold used.
    """
    artifact = artifact or load_artifact()
    scored = predict_batch(pd.DataFrame([record]), artifact)
    proba = float(scored["churn_probability"].iloc[0])
    return {
        "churn_probability": proba,
        "churn_prediction": int(scored["churn_prediction"].iloc[0]),
        "risk_band": _risk_band(proba),
        "threshold": float(artifact["threshold"]),
    }


def _risk_band(probability: float) -> str:
    """Map a churn probability to a Low/Medium/High action band."""
    for cutoff, label in RISK_BANDS:
        if probability >= cutoff:
            return label
    return "Low"
