"""
Author : Mohit Patle

Description:
End-to-end training pipeline. Reproduces the entire project from the
raw Excel file to a serialized model artifact in one command:

    python -m src.train_pipeline

Stages: load -> validate -> clean -> engineer features -> split ->
compare models (CV) -> tune best -> tune threshold -> evaluate on the
held-out test set -> save plots, metrics, the drift reference profile and
the versioned model artifact -> log the run to MLflow.
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from src.monitoring import build_reference_profile, save_reference_profile
from src.preprocessing import clean_raw_data, split_data
from src.tracking import log_mlflow_run, make_model_version, register_model
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

    logger.info("=== 1/9 Load raw data ===")
    df_raw = load_data(str(root / config["data"]["raw_path"]))

    logger.info("=== 2/9 Validate ===")
    validation_report = validate_raw_data(df_raw, target=config["columns"]["target"])

    logger.info("=== 3/9 Clean + engineer features ===")
    df = add_features(clean_raw_data(df_raw, config))

    logger.info("=== 4/9 Split and persist processed data ===")
    X_train, X_test, y_train, y_test = split_data(df, config)
    X_train.assign(churn_value=y_train).to_csv(root / config["data"]["train_path"], index=False)
    X_test.assign(churn_value=y_test).to_csv(root / config["data"]["test_path"], index=False)

    logger.info("=== 5/9 Cross-validated model comparison ===")
    numeric, categorical = get_feature_lists(X_train)
    preprocessor = build_preprocessor(numeric, categorical)
    models = get_models(y_train, config["training"]["random_state"])
    comparison = compare_models(models, preprocessor, X_train, y_train, config)
    comparison.to_csv(root / "reports" / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    logger.info("Best model family by CV ROC-AUC: %s", best_name)

    logger.info("=== 6/9 Hyperparameter tuning (%s) ===", best_name)
    pipeline = build_pipeline(preprocessor, models[best_name])
    search = tune_model(
        pipeline, get_param_distributions(best_name), X_train, y_train, config
    )
    tuned = search.best_estimator_

    logger.info("=== 7/9 Threshold tuning + test evaluation ===")
    threshold = find_best_threshold(tuned, X_train, y_train, config)
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    version = make_model_version(trained_at)
    metrics: dict[str, Any] = {
        "model": best_name,
        "model_version": version,
        "trained_at": trained_at,
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

    logger.info("=== 8/9 Serialize, version and register model ===")
    # Drift reference: what "normal" model inputs looked like at training time.
    profile_path = root / config["monitoring"]["reference_profile_path"]
    save_reference_profile(
        build_reference_profile(X_train, config["monitoring"]["psi_bins"]), profile_path
    )

    artifact = {
        "model": tuned,
        "model_name": best_name,
        "threshold": threshold,
        "input_columns": list(X_train.columns),
        "metrics": metrics,
        "trained_at": metrics["trained_at"],
        "model_version": version,
        "sklearn_version": sklearn.__version__,
    }
    model_path = root / config["artifacts"]["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    logger.info("Saved model artifact to %s", model_path)
    register_model(model_path, version, metrics, config)

    logger.info("=== 9/9 Log experiment to MLflow ===")
    tuned_metrics = metrics["test_at_tuned_threshold"]
    log_mlflow_run(
        config,
        params={
            "model": best_name,
            **{k.removeprefix("model__"): v for k, v in metrics["best_params"].items()},
            "threshold": round(threshold, 4),
            "cv_folds": config["training"]["cv_folds"],
        },
        metrics={
            "cv_roc_auc": metrics["cv_roc_auc"],
            **{f"test_{k}": float(v) for k, v in tuned_metrics.items()
               if k not in ("threshold", "confusion_matrix")},
        },
        artifact_paths=[
            root / config["artifacts"]["metrics_path"],
            root / "reports" / "model_comparison.csv",
            root / "reports" / "feature_importance.csv",
            profile_path,
            *sorted(images_dir.glob("*.png")),
        ],
        run_name=version,
    )

    return metrics


if __name__ == "__main__":
    main()
