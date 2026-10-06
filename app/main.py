from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "outputs/model_bundle.joblib"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    run_id: str


def _load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    required = {"run_id", "model_version", "model_name", "feature_columns", "preprocessor", "model"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle missing required keys: {sorted(missing)}")
    return bundle


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.bundle = None
    application.state.load_error = None
    try:
        application.state.bundle = _load_bundle()
    except Exception as exc:  # liveness remains available while readiness reports the model error
        application.state.load_error = str(exc)
    yield


app = FastAPI(title="Airbnb Guest Demand Proxy API", version="2.0.0", lifespan=lifespan)


def _require_bundle(request: Request) -> dict[str, Any]:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error or "Model not loaded")
    return bundle


def _validate_features(features: dict[str, float], expected_features: list[str]) -> pd.DataFrame:
    received = set(features)
    missing = [name for name in expected_features if name not in received]
    unexpected = sorted(received - set(expected_features))
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": unexpected},
        )

    values = []
    for name in expected_features:
        value = float(features[name])
        if not math.isfinite(value):
            raise HTTPException(status_code=422, detail=f"Feature '{name}' must be finite.")
        values.append(value)
    return pd.DataFrame([values], columns=expected_features)


def _score_one(payload: PredictionRequest, bundle: dict[str, Any]) -> PredictionResponse:
    expected = list(bundle["feature_columns"])
    frame = _validate_features(payload.features, expected)
    transformed = bundle["preprocessor"].transform(frame)
    transformed_frame = pd.DataFrame(transformed, columns=expected)
    prediction = float(bundle["model"].predict(transformed_frame)[0])
    return PredictionResponse(
        prediction=prediction,
        model_name=bundle["model_name"],
        model_version=bundle["model_version"],
        run_id=bundle["run_id"],
    )


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    """Backward-compatible liveness alias."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    bundle = _require_bundle(request)
    return {"status": "ready", "model_version": bundle["model_version"]}


@app.get("/metadata")
def metadata(request: Request) -> dict[str, Any]:
    bundle = _require_bundle(request)
    return {
        "run_id": bundle["run_id"],
        "model_version": bundle["model_version"],
        "model_name": bundle["model_name"],
        "feature_columns": bundle["feature_columns"],
        "target_column": bundle.get("training_config", {}).get("target_column", "target"),
        "source_metadata": bundle.get("source_metadata", {}),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    return _score_one(payload, _require_bundle(request))


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(payloads: list[PredictionRequest], request: Request) -> list[PredictionResponse]:
    if not payloads:
        raise HTTPException(status_code=400, detail="At least one record is required.")
    if len(payloads) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Batch exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}.")
    bundle = _require_bundle(request)
    return [_score_one(payload, bundle) for payload in payloads]
