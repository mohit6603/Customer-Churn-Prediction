"""
Author : Mohit Patle

Description:
Model definitions, cross-validated comparison and hyperparameter
tuning. Every model is wrapped in a Pipeline together with the
preprocessor, so all fitted transforms happen inside each CV fold —
the only structurally safe way to cross-validate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.utils import get_logger

logger = get_logger(__name__)

SCORING = {
    "roc_auc": "roc_auc",
    "avg_precision": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def get_models(y_train: pd.Series, random_state: int = 42) -> dict[str, BaseEstimator]:
    """
    Return the candidate models, configured for class imbalance.

    ``class_weight='balanced'`` (sklearn) and ``scale_pos_weight``
    (XGBoost) up-weight the minority churn class instead of resampling,
    which keeps the pipeline simple and leakage-free.

    Parameters
    ----------
    y_train : pd.Series
        Training target, used to compute the positive-class weight.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict[str, BaseEstimator]
        Mapping of model name -> unfitted estimator.
    """
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_pipeline(preprocessor: ColumnTransformer, model: BaseEstimator) -> Pipeline:
    """
    Combine preprocessor and model into a single sklearn Pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Unfitted preprocessing stage.
    model : BaseEstimator
        Unfitted estimator.

    Returns
    -------
    Pipeline
        Two-step pipeline: 'preprocessor' -> 'model'.
    """
    return Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(model))])


def compare_models(
    models: dict[str, BaseEstimator],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """
    Cross-validate every candidate model and rank them.

    Parameters
    ----------
    models : dict[str, BaseEstimator]
        Candidates from ``get_models``.
    preprocessor : ColumnTransformer
        Shared preprocessing stage (cloned per pipeline).
    X_train : pd.DataFrame
        Training features (raw, pre-transform).
    y_train : pd.Series
        Training target.
    config : dict
        Project configuration with the ``training`` section.

    Returns
    -------
    pd.DataFrame
        One row per model with mean CV scores, sorted by ROC-AUC.
    """
    train_cfg = config["training"]
    cv = StratifiedKFold(
        n_splits=train_cfg["cv_folds"], shuffle=True,
        random_state=train_cfg["random_state"],
    )

    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        pipeline = build_pipeline(preprocessor, model)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name}
        row.update(
            {metric: float(np.mean(scores[f"test_{metric}"])) for metric in SCORING}
        )
        rows.append(row)
        logger.info("CV %s: ROC-AUC=%.4f recall=%.4f", name, row["roc_auc"], row["recall"])

    return (
        pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    )


def get_param_distributions(model_name: str) -> dict[str, Any]:
    """
    Hyperparameter search space for a given model family.

    Parameters
    ----------
    model_name : str
        One of 'logistic_regression', 'random_forest', 'xgboost'.

    Returns
    -------
    dict[str, Any]
        Param distributions keyed with the 'model__' pipeline prefix.

    Raises
    ------
    KeyError
        If the model name has no defined search space.
    """
    spaces: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "model__C": loguniform(1e-3, 1e2),
        },
        "random_forest": {
            "model__n_estimators": randint(200, 800),
            "model__max_depth": randint(4, 20),
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "xgboost": {
            "model__n_estimators": randint(200, 800),
            "model__max_depth": randint(3, 8),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": loguniform(0.1, 10),
        },
    }
    if model_name not in spaces:
        raise KeyError(f"No parameter space defined for '{model_name}'.")
    return spaces[model_name]


def tune_model(
    pipeline: Pipeline,
    param_distributions: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> RandomizedSearchCV:
    """
    Randomised hyperparameter search over a pipeline.

    Randomised (rather than grid) search explores wide spaces at a
    fixed compute budget, which almost always finds better models per
    CPU-hour on tabular problems.

    Parameters
    ----------
    pipeline : Pipeline
        Preprocessor + model pipeline to tune.
    param_distributions : dict[str, Any]
        Search space from ``get_param_distributions``.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    config : dict
        Project configuration with the ``training`` section.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object; ``.best_estimator_`` is refit on all of
        X_train with the winning parameters.
    """
    train_cfg = config["training"]
    cv = StratifiedKFold(
        n_splits=train_cfg["cv_folds"], shuffle=True,
        random_state=train_cfg["random_state"],
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=train_cfg["n_iter"],
        scoring=train_cfg["scoring"],
        cv=cv,
        random_state=train_cfg["random_state"],
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    logger.info("Best CV %s=%.4f with params: %s",
                train_cfg["scoring"], search.best_score_, search.best_params_)
    return search
