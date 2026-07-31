"""
Author : Mohit Patle

Description:
End-to-end training pipeline. Reproduces the entire project from the
raw Excel file to a serialized model artifact in one command:

    python -m src.train_pipeline

Stages: load -> validate -> clean -> engineer features -> split ->
compare models (CV) -> tune best -> tune threshold -> evaluate on the
held-out test set -> save plots, metrics and the model artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: the pipeline runs without a display

import joblib
import sklearn

from src.data_loader import load_data
from src.data_validation import validate_raw_data
from src.evaluation import (
    compute_feature_importance,
    evaluate_model,
    find_best_threshold,
    plot_evaluation_curves,
    plot_feature_importance,
    save_metrics,
)
from src.feature_engineering import add_features, build_preprocessor, get_feature_lists
from src.model_training import (
    build_pipeline,
    compare_models,
    get_models,
    get_param_distributions,
    tune_model,
)
from src.preprocessing import clean_raw_data, split_data
from src.utils import get_logger, get_project_root, load_config

logger = get_logger(__name__)


def _to_native(obj: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy scalars in a dict to native Python for JSON."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in obj.items()}


def main() -> dict[str, Any]:
    """
    Run the full training pipeline and persist all artifacts.

    Returns
    -------
    dict[str, Any]
        Final metrics summary (also written to reports/metrics.json).
    """
    config = load_config()
    root = get_project_root()

    logger.info("=== 1/8 Load raw data ===")
    df_raw = load_data(str(root / config["data"]["raw_path"]))

    logger.info("=== 2/8 Validate ===")
    validation_report = validate_raw_data(df_raw, target=config["columns"]["target"])

    logger.info("=== 3/8 Clean + engineer features ===")
    df = add_features(clean_raw_data(df_raw, config))

    logger.info("=== 4/8 Split and persist processed data ===")
    X_train, X_test, y_train, y_test = split_data(df, config)
    X_train.assign(churn_value=y_train).to_csv(root / config["data"]["train_path"], index=False)
    X_test.assign(churn_value=y_test).to_csv(root / config["data"]["test_path"], index=False)

    logger.info("=== 5/8 Cross-validated model comparison ===")
    numeric, categorical = get_feature_lists(X_train)
    preprocessor = build_preprocessor(numeric, categorical)
    models = get_models(y_train, config["training"]["random_state"])
    comparison = compare_models(models, preprocessor, X_train, y_train, config)
    comparison.to_csv(root / "reports" / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    logger.info("Best model family by CV ROC-AUC: %s", best_name)

    logger.info("=== 6/8 Hyperparameter tuning (%s) ===", best_name)
    pipeline = build_pipeline(preprocessor, models[best_name])
    search = tune_model(
        pipeline, get_param_distributions(best_name), X_train, y_train, config
    )
    tuned = search.best_estimator_

    logger.info("=== 7/8 Threshold tuning + test evaluation ===")
    threshold = find_best_threshold(tuned, X_train, y_train, config)
    metrics: dict[str, Any] = {
        "model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_params": _to_native(search.best_params_),
        "cv_roc_auc": float(search.best_score_),
        "validation_report": validation_report,
        "test_at_default_threshold": evaluate_model(tuned, X_test, y_test, 0.5),
        "test_at_tuned_threshold": evaluate_model(tuned, X_test, y_test, threshold),
    }

    images_dir = root / config["artifacts"]["images_dir"]
    plot_evaluation_curves(tuned, X_test, y_test, threshold, images_dir)
    importance = compute_feature_importance(tuned, X_test, y_test)
    importance.to_csv(root / "reports" / "feature_importance.csv", index=False)
    plot_feature_importance(importance, images_dir)
    save_metrics(metrics, root / config["artifacts"]["metrics_path"])

    logger.info("=== 8/8 Serialize model artifact ===")
    artifact = {
        "model": tuned,
        "model_name": best_name,
        "threshold": threshold,
        "input_columns": list(X_train.columns),
        "metrics": metrics,
        "trained_at": metrics["trained_at"],
        "sklearn_version": sklearn.__version__,
    }
    model_path = root / config["artifacts"]["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    logger.info("Saved model artifact to %s", model_path)

    return metrics


if __name__ == "__main__":
    main()
