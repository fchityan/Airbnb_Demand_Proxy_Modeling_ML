from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    run_id: str


def _load_bundle() -> dict[str, Any]:
    model_path = Path(os.getenv("MODEL_PATH", "outputs/model_bundle.joblib"))
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    return joblib.load(model_path)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.bundle = _load_bundle()
    yield


app = FastAPI(
    title="Airbnb Demand Proxy API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "bundle"):
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}


@app.get("/metadata")
def metadata(request: Request) -> dict[str, Any]:
    bundle = request.app.state.bundle
    return {
        "run_id": bundle["run_id"],
        "model_version": bundle["model_version"],
        "model_name": bundle["model_name"],
        "feature_columns": bundle["feature_columns"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    bundle = request.app.state.bundle
    expected_features = list(bundle["feature_columns"])
    received_features = set(payload.features)
    missing_features = [name for name in expected_features if name not in received_features]
    unexpected_features = sorted(received_features - set(expected_features))
    if missing_features or unexpected_features:
        detail = {
            "missing_features": missing_features,
            "unexpected_features": unexpected_features,
        }
        raise HTTPException(status_code=422, detail=detail)

    feature_frame = pd.DataFrame(
        [[payload.features[name] for name in expected_features]],
        columns=expected_features,
    )
    transformed_features = pd.DataFrame(
        bundle["preprocessor"].transform(feature_frame),
        columns=expected_features,
    )
    prediction = float(bundle["model"].predict(transformed_features)[0])
    return PredictionResponse(
        prediction=prediction,
        model_name=bundle["model_name"],
        model_version=bundle["model_version"],
        run_id=bundle["run_id"],
    )