"""
Author : Mohit Patle

Description:
FastAPI service that serves the churn model in production.

Endpoints
---------
GET  /health              liveness + which model version is loaded
GET  /model/info          training metrics, threshold and input contract
POST /predict             score one customer (JSON)
POST /predict/batch       score a CSV upload, ranked highest risk first
GET  /monitoring/stats    serving statistics from the prediction log
GET  /monitoring/drift    PSI drift of recent traffic vs. training data

Every prediction is written to the SQLite prediction log, which is what
/monitoring/* reads from.

Run from the project root:
    uvicorn api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import RedirectResponse

from api.schemas import (
    BatchPredictionOut,
    CustomerIn,
    DriftOut,
    HealthOut,
    ModelInfoOut,
    PredictionOut,
)
from src.inference import get_model_version, load_artifact, predict_batch, prepare_features
from src.monitoring import (
    PredictionLogger,
    detect_drift,
    load_reference_profile,
    summarize_drift,
)
from src.utils import get_logger, get_project_root, load_config

logger = get_logger(__name__, log_file="api.log")

# What the retention team should do for each risk band.
RECOMMENDED_ACTIONS: dict[str, str] = {
    "High": "Immediate retention call: offer a contract upgrade or targeted discount.",
    "Medium": "Proactive check-in or a service add-on offer.",
    "Low": "No action needed: customer profile looks stable.",
}

# PSI on a handful of rows is noise, not drift.
MIN_DRIFT_SAMPLES = 30


def create_app(
    artifact: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
    reference_profile: dict[str, Any] | None = None,
) -> FastAPI:
    """
    Build the FastAPI application.

    Parameters
    ----------
    artifact : dict[str, Any] | None
        Model artifact; loaded from models/ at startup when omitted.
        Tests inject a tiny in-memory model here.
    log_path : str | Path | None
        SQLite prediction log; defaults to the path in the config.
    reference_profile : dict[str, Any] | None
        Drift reference profile; loaded from reports/ when omitted.

    Returns
    -------
    FastAPI
        The configured application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = load_config()
        app.state.config = config
        app.state.artifact = artifact or load_artifact()
        app.state.model_version = get_model_version(app.state.artifact)
        app.state.prediction_log = PredictionLogger(
            log_path or get_project_root() / config["monitoring"]["prediction_log_path"]
        )
        app.state.reference_profile = reference_profile
        logger.info("API ready with model %s", app.state.model_version)
        yield

    app = FastAPI(
        title="Customer Churn Prediction API",
        description="Scores telecom customers for churn risk and logs every "
                    "prediction for drift monitoring.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path,
                    response.status_code, elapsed_ms)
        return response

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthOut, tags=["ops"])
    def health(request: Request) -> HealthOut:
        return HealthOut(status="ok", model_loaded=True,
                         model_version=request.app.state.model_version)

    @app.get("/model/info", response_model=ModelInfoOut, tags=["ops"])
    def model_info(request: Request) -> ModelInfoOut:
        art = request.app.state.artifact
        metrics = art.get("metrics", {})
        tuned = metrics.get("test_at_tuned_threshold", {})
        return ModelInfoOut(
            model_name=art.get("model_name", "unknown"),
            model_version=request.app.state.model_version,
            trained_at=str(art.get("trained_at", "unknown")),
            threshold=round(float(art["threshold"]), 4),
            cv_roc_auc=metrics.get("cv_roc_auc"),
            test_metrics={k: float(v) for k, v in tuned.items()
                          if isinstance(v, (int, float))},
            input_columns=art["input_columns"],
        )

    @app.post("/predict", response_model=PredictionOut, tags=["predict"])
    def predict(customer: CustomerIn, request: Request) -> PredictionOut:
        df = pd.DataFrame([customer.to_raw_record()])
        return _score(df, request.app)[0]

    @app.post("/predict/batch", response_model=BatchPredictionOut, tags=["predict"])
    async def predict_file(
        request: Request,
        file: UploadFile = File(description="CSV in the raw IBM export schema"),
        top_n: int | None = Query(default=None, ge=1,
                                  description="Return only the N highest-risk customers"),
    ) -> BatchPredictionOut:
        try:
            df = pd.read_csv(io.BytesIO(await file.read()))
        except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
            raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

        predictions = _score(df, request.app)
        return BatchPredictionOut(
            model_version=request.app.state.model_version,
            n_customers=len(predictions),
            n_flagged=sum(p.churn_prediction for p in predictions),
            predictions=predictions[:top_n] if top_n else predictions,
        )

    @app.get("/monitoring/stats", tags=["monitoring"])
    def monitoring_stats(request: Request) -> dict[str, Any]:
        return request.app.state.prediction_log.stats()

    @app.get("/monitoring/drift", response_model=DriftOut, tags=["monitoring"])
    def monitoring_drift(
        request: Request,
        limit: int = Query(default=1000, ge=MIN_DRIFT_SAMPLES,
                           description="Compare the N most recent predictions"),
    ) -> DriftOut:
        state = request.app.state
        if state.reference_profile is None:
            try:
                state.reference_profile = load_reference_profile()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        inputs = state.prediction_log.load_inputs(limit)
        if len(inputs) < MIN_DRIFT_SAMPLES:
            raise HTTPException(
                status_code=409,
                detail=f"Need at least {MIN_DRIFT_SAMPLES} logged predictions "
                       f"for a drift check; have {len(inputs)}.",
            )
        features = prepare_features(inputs, state.config)
        summary = summarize_drift(detect_drift(state.reference_profile, features))
        return DriftOut(n_samples=len(inputs), **summary)

    return app


def _score(df: pd.DataFrame, app: FastAPI) -> list[PredictionOut]:
    """Score raw-schema rows, log them, and shape the API response."""
    state = app.state
    try:
        scored = predict_batch(df, state.artifact)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    state.prediction_log.log(df, scored, state.model_version)

    threshold = round(float(state.artifact["threshold"]), 4)
    ids = scored["CustomerID"] if "CustomerID" in scored else pd.Series(None, index=scored.index)
    return [
        PredictionOut(
            customer_id=None if pd.isna(cid) else str(cid),
            churn_probability=round(float(proba), 4),
            churn_prediction=int(pred),
            risk_band=band,
            recommended_action=RECOMMENDED_ACTIONS[band],
            threshold=threshold,
            model_version=state.model_version,
        )
        for cid, proba, pred, band in zip(
            ids, scored["churn_probability"], scored["churn_prediction"], scored["risk_band"],
            strict=True,
        )
    ]


app = create_app()
