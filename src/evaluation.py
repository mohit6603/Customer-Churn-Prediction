"""
Author : Mohit Patle

Description:
Model evaluation: threshold tuning, test-set metrics, diagnostic plots
and model interpretation via permutation importance.

The decision threshold is tuned on OUT-OF-FOLD predictions from the
training set (never on the test set), because the threshold is a model
parameter like any other — choosing it on test data would leak.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.utils import get_logger

logger = get_logger(__name__)


def find_best_threshold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    beta: float = 2.0,
) -> float:
    """
    Choose the decision threshold that maximises F-beta on
    out-of-fold training predictions.

    Beta=2 weights recall twice as much as precision, matching the
    business economics: a missed churner (lost lifetime value) costs
    far more than a wasted retention offer.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted (or clone-able) pipeline with best hyperparameters.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    config : dict
        Project configuration with the ``training`` section.
    beta : float
        Recall weight in the F-beta score.

    Returns
    -------
    float
        Best probability threshold in [0.05, 0.95].
    """
    train_cfg = config["training"]
    cv = StratifiedKFold(
        n_splits=train_cfg["cv_folds"], shuffle=True,
        random_state=train_cfg["random_state"],
    )
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [
        fbeta_score(y_train, (oof_proba >= t).astype(int), beta=beta)
        for t in thresholds
    ]
    best_t = float(thresholds[int(np.argmax(scores))])

    logger.info("Best threshold=%.2f (F%.0f=%.4f on out-of-fold predictions)",
                best_t, beta, max(scores))
    return best_t


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute test-set metrics at a given decision threshold.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    X_test : pd.DataFrame
        Test features (raw, pre-transform).
    y_test : pd.Series
        Test target.
    threshold : float
        Probability cut-off for the positive (churn) class.

    Returns
    -------
    dict[str, Any]
        Metric name -> value, including the confusion matrix.
    """
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "f2": float(fbeta_score(y_test, pred, beta=2.0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    logger.info("Test metrics @%.2f: %s", threshold,
                {k: v for k, v in metrics.items() if k != "confusion_matrix"})
    return metrics


def plot_evaluation_curves(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    images_dir: str | Path,
) -> None:
    """
    Save confusion-matrix, ROC and precision-recall plots.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    threshold : float
        Decision threshold used for the confusion matrix.
    images_dir : str | Path
        Output directory for the PNG files.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred), display_labels=["Stayed", "Churned"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(images_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", linewidth=0.8)
    axes[1].set_title("Precision-Recall curve")
    fig.tight_layout()
    fig.savefig(images_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    logger.info("Saved evaluation plots to %s", images_dir)


def compute_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance of the ORIGINAL input features.

    Running permutation on the whole pipeline (raw columns in, score
    out) attributes importance to business-level features rather than
    to individual one-hot dummy columns, which is what stakeholders
    and interviewers actually want to see.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    n_repeats : int
        Permutation repeats per feature (more = stabler estimates).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance_mean, importance_std;
        sorted descending by importance_mean.
    """
    result = permutation_importance(
        pipeline, X_test, y_test,
        scoring="roc_auc", n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    importance_df: pd.DataFrame, images_dir: str | Path, top_n: int = 15
) -> None:
    """
    Save a horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of ``compute_feature_importance``.
    images_dir : str | Path
        Output directory for the PNG file.
    top_n : int
        Number of features to display.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Permutation importance (drop in ROC-AUC)")
    ax.set_title(f"Top {top_n} features")
    fig.tight_layout()
    fig.savefig(images_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    logger.info("Saved feature-importance plot to %s", images_dir)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """
    Persist a metrics dictionary as pretty-printed JSON.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics to save.
    path : str | Path
        Output JSON file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics to %s", path)
