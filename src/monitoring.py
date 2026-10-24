"""
Author : Mohit Patle

Description:
Production monitoring for the churn model. Two responsibilities:

1. Data drift — a reference profile of every model input is captured at
   training time; new data is compared against it with the Population
   Stability Index (PSI). A model is only as good as the assumption that
   tomorrow's customers look like the ones it was trained on.
2. Prediction logging — every prediction served by the API is written
   to SQLite (inputs, score, model version, timestamp), so drift and
   score distributions can be audited after the fact.

Check a new customer file for drift from the command line:

    python -m src.monitoring path/to/new_customers.csv
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference import prepare_features
from src.utils import get_logger, get_project_root, load_config

logger = get_logger(__name__)

# Industry-standard PSI cut-offs.
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25

# Floor for empty bins so log(actual / expected) stays finite.
_EPS = 1e-4
_OTHER = "__other__"


# --------------------------------------------------------------------------
# Data drift
# --------------------------------------------------------------------------


def build_reference_profile(X: pd.DataFrame, n_bins: int = 10) -> dict[str, Any]:
    """
    Summarise the training distribution of every feature.

    Numeric features are binned on training quantiles; categorical
    features keep their category frequencies. Only aggregates are
    stored — no customer-level data — so the profile is safe to commit.

    Parameters
    ----------
    X : pd.DataFrame
        Training features (after cleaning + feature engineering).
    n_bins : int
        Number of quantile bins for numeric features.

    Returns
    -------
    dict[str, Any]
        {feature: {"type": "numeric", "edges": [...], "proportions": [...]}
         or {"type": "categorical", "proportions": {category: share}}}
    """
    profile: dict[str, Any] = {}
    for col in X.columns:
        series = X[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            # Inner edges only; the outer bins are open-ended so values
            # outside the training range still land in a bin.
            quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
            edges = sorted(set(np.quantile(series, quantiles).round(6).tolist()))
            profile[col] = {
                "type": "numeric",
                "edges": edges,
                "proportions": _numeric_proportions(series, edges).tolist(),
            }
        else:
            shares = series.astype(str).value_counts(normalize=True)
            profile[col] = {
                "type": "categorical",
                "proportions": {str(k): float(v) for k, v in shares.items()},
            }
    return profile


def compute_psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """
    Population Stability Index between two binned distributions.

    PSI = sum((actual - expected) * ln(actual / expected))

    Parameters
    ----------
    expected : np.ndarray
        Bin proportions in the reference (training) data.
    actual : np.ndarray
        Bin proportions in the new data, same bins.

    Returns
    -------
    float
        PSI value: < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 significant.
    """
    expected = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def detect_drift(profile: dict[str, Any], X_new: pd.DataFrame) -> pd.DataFrame:
    """
    Compare new data against the training reference profile.

    Parameters
    ----------
    profile : dict[str, Any]
        Output of ``build_reference_profile``.
    X_new : pd.DataFrame
        New features (after cleaning + feature engineering).

    Returns
    -------
    pd.DataFrame
        One row per feature with columns feature, psi and status
        (stable / moderate / significant), most drifted first.
    """
    rows = []
    for col, ref in profile.items():
        if col not in X_new.columns:
            continue
        series = X_new[col].dropna()

        if ref["type"] == "numeric":
            expected = np.asarray(ref["proportions"])
            actual = _numeric_proportions(pd.to_numeric(series), ref["edges"])
        else:
            categories = list(ref["proportions"])
            shares = series.astype(str).value_counts(normalize=True)
            # Categories never seen in training are pooled into one bin.
            expected = np.array([*ref["proportions"].values(), 0.0])
            actual = np.array([
                *(shares.get(c, 0.0) for c in categories),
                shares[~shares.index.isin(categories)].sum(),
            ])

        psi = compute_psi(expected, actual)
        rows.append({"feature": col, "psi": round(psi, 4), "status": _psi_status(psi)})

    report = pd.DataFrame(rows, columns=["feature", "psi", "status"])
    return report.sort_values("psi", ascending=False, ignore_index=True)


def summarize_drift(report: pd.DataFrame) -> dict[str, Any]:
    """Collapse a per-feature drift report into a single verdict."""
    drifted = report.loc[report["status"] == "significant", "feature"].tolist()
    moderate = report.loc[report["status"] == "moderate", "feature"].tolist()
    return {
        "drift_detected": bool(drifted),
        "retraining_recommended": bool(drifted),
        "significant_features": drifted,
        "moderate_features": moderate,
        "features": report.to_dict(orient="records"),
    }


def save_reference_profile(profile: dict[str, Any], path: str | Path) -> None:
    """Persist the reference profile as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Saved reference profile (%d features) to %s", len(profile), path)


def load_reference_profile(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the reference profile written by the training pipeline.

    Raises
    ------
    FileNotFoundError
        If the pipeline has not been run since monitoring was added.
    """
    if path is None:
        path = get_project_root() / load_config()["monitoring"]["reference_profile_path"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No reference profile at {path}. Run 'python -m src.train_pipeline' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_proportions(series: pd.Series, edges: list[float]) -> np.ndarray:
    """Share of values falling in each open-ended bin defined by edges."""
    if len(series) == 0:
        return np.zeros(len(edges) + 1)
    counts = np.bincount(np.searchsorted(edges, series, side="right"),
                         minlength=len(edges) + 1)
    return counts / counts.sum()


def _psi_status(psi: float) -> str:
    """Map a PSI value to a stability label."""
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


# --------------------------------------------------------------------------
# Prediction logging
# --------------------------------------------------------------------------


class PredictionLogger:
    """
    Append-only SQLite log of every prediction the API serves.

    Parameters
    ----------
    db_path : str | Path
        SQLite file; created (with its parent directory) on first use.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at         TEXT    NOT NULL,
            model_version     TEXT    NOT NULL,
            customer_id       TEXT,
            churn_probability REAL    NOT NULL,
            churn_prediction  INTEGER NOT NULL,
            risk_band         TEXT    NOT NULL,
            features          TEXT    NOT NULL
        )
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(self._SCHEMA)

    def log(self, inputs: pd.DataFrame, scored: pd.DataFrame, model_version: str) -> None:
        """
        Record a batch of predictions.

        Parameters
        ----------
        inputs : pd.DataFrame
            Raw-schema rows exactly as received.
        scored : pd.DataFrame
            Output of ``predict_batch`` (same index as inputs).
        model_version : str
            Identifier of the model that produced the scores.
        """
        now = datetime.now(UTC).isoformat(timespec="seconds")
        rows = [
            (
                now,
                model_version,
                None if pd.isna(record.get("CustomerID")) else str(record.get("CustomerID")),
                float(scored.at[idx, "churn_probability"]),
                int(scored.at[idx, "churn_prediction"]),
                str(scored.at[idx, "risk_band"]),
                json.dumps(record, default=str),
            )
            for idx, record in zip(inputs.index, inputs.to_dict(orient="records"), strict=True)
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO predictions (logged_at, model_version, customer_id, "
                "churn_probability, churn_prediction, risk_band, features) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def load(self, limit: int | None = None) -> pd.DataFrame:
        """Return logged predictions, newest first."""
        query = "SELECT * FROM predictions ORDER BY id DESC"
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def load_inputs(self, limit: int | None = None) -> pd.DataFrame:
        """Return the raw-schema inputs of logged predictions."""
        logged = self.load(limit)
        return pd.DataFrame([json.loads(f) for f in logged["features"]])

    def stats(self) -> dict[str, Any]:
        """Aggregate serving statistics for the monitoring endpoint."""
        logged = self.load()
        if logged.empty:
            return {"n_predictions": 0}
        return {
            "n_predictions": len(logged),
            "first_logged_at": logged["logged_at"].min(),
            "last_logged_at": logged["logged_at"].max(),
            "mean_churn_probability": round(float(logged["churn_probability"].mean()), 4),
            "flagged_rate": round(float(logged["churn_prediction"].mean()), 4),
            "risk_band_counts": logged["risk_band"].value_counts().to_dict(),
            "model_versions": logged["model_version"].value_counts().to_dict(),
        }

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    """Score a CSV for drift against the training reference profile."""
    parser = argparse.ArgumentParser(description="Check a customer file for data drift.")
    parser.add_argument("csv", help="Customer CSV in the raw export schema")
    parser.add_argument("--out", default=None, help="Where to write the JSON report")
    args = parser.parse_args()

    config = load_config()
    new_features = prepare_features(pd.read_csv(args.csv), config)
    report = detect_drift(load_reference_profile(), new_features)
    summary = summarize_drift(report)

    default_out = get_project_root() / config["monitoring"]["drift_report_path"]
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report.to_string(index=False))
    verdict = ("DRIFT DETECTED, retraining recommended" if summary["drift_detected"]
               else "No significant drift")
    print(f"\n{verdict}. Report written to {out}")


if __name__ == "__main__":
    main()
