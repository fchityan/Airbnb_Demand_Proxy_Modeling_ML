# Airbnb Guest Demand Proxy — Production ML

A production-oriented regression system based on the original Airbnb demand-proxy notebook, with explicit target semantics, validation-only model selection, baseline promotion gates, versioned artifacts, monitoring, and online/batch serving.

## Portfolio Snapshot

**Problem:** Search and contact data contain imperfect demand signals, and a useful production model must separate feature construction, model selection, locked evaluation, deployment, and monitoring.

**Notebook decision retained:** The original analysis selected **guest count (`n_guests`)** as the final regression target after the price-filter target proved too noisy. The production pipeline now enforces that decision directly.

**Production output:** A leakage-controlled feature frame, train/validation/test model lifecycle, mean-baseline promotion gate, Linear Regression/XGBoost comparison, immutable run artifacts, versioned model bundle, artifact fingerprint, drift reports, audit log, retention metadata, and FastAPI online/batch inference.

**Stack:** Python · pandas · scikit-learn · XGBoost · FastAPI · Docker · MLOps

## Important production corrections

The production upgrade fixes several gaps that matter more than adding another model:

1. **Target alignment:** workbook-based runs now predict `n_guests`, matching the notebook's final modeling experiment.
2. **Leakage prevention:** guest count is never included as an input feature. Identifiers and contact-message counts are also excluded from the production feature set.
3. **Fail-closed ingestion:** if a production data path is supplied but missing or malformed, training now fails instead of silently switching to synthetic data.
4. **Validation-only model selection:** model choice is made on a validation split, not the final holdout test set.
5. **Baseline promotion gate:** the selected ML model is compared with a mean baseline before promotion status is assigned.
6. **Locked test evaluation:** the test split is used only after the model family has been selected.

## Workbook production features

For the raw `contacts.xlsx` + `searches.xlsx` workflow, the model frame uses search-intent features available before the target is evaluated:

```text
n_searches
n_nights
filter_price_min
filter_price_max
price_midpoint
price_range
room_type_filter_count
neighborhood_filter_count
has_room_type_filter
has_neighborhood_filter
```

Target:

```text
n_guests -> target
```

The loader deliberately does not train on IDs, the guest-count target itself, or `n_messages`.

## Data modes

### Production workbook directory

Place these files in one directory:

```text
contacts.xlsx
searches.xlsx
```

Then run:

```python
from pathlib import Path
from src.run_pipeline import run_pipeline

run_pipeline(
    output_dir=Path("outputs"),
    data_path="data",
    source_name="airbnb_workbooks",
    source_version="v1",
)
```

### Model-ready CSV / Parquet

A model-ready external table must contain numeric features plus either `target` or `n_guests`. If `n_guests` is supplied, the loader renames it to `target` and removes it from the feature set.

### Synthetic development mode

```bash
python -m src.run_pipeline
```

Synthetic data is generated **only when no `data_path` is supplied**. It exists for pipeline tests and local engineering, not as a substitute for missing production data.

## Model lifecycle

```text
source data
   │
   ▼
schema + numeric validation
   │
   ▼
60% train / 20% validation / 20% test
   │
   ├── mean baseline
   ├── linear regression
   └── XGBoost
   │
   ▼
validation RMSE model selection
   │
   ▼
baseline promotion gate
   │
   ▼
refit selected model on train + validation
   │
   ▼
locked test evaluation
   │
   ├── drift / prediction-shift monitoring
   ├── run manifest + SHA-256 artifact fingerprint
   └── model_bundle.joblib
```

The model bundle contains the fitted scaler, selected model, feature contract, run ID, model version, source lineage, training configuration, validation result, test result, and promotion status.

## Outputs

Each run writes immutable artifacts under:

```text
outputs/runs/<run_id>/
```

Key artifacts:

```text
validation_model_comparison.csv
test_metrics.csv
feature_importance.csv
drift_report.csv
prediction_shift.csv
monitoring_alerts.json
summary.json
run_manifest.json
model_bundle_<run_id>.joblib
```

Canonical latest artifacts are also copied to `outputs/` for deployment and review.

The run manifest includes the SHA-256 fingerprint of the versioned model artifact so deployment automation can verify artifact integrity.

## Serving API

The API loads the bundle once at startup and exposes separate process-health and model-readiness semantics.

```bash
MODEL_PATH=outputs/model_bundle.joblib uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

```text
GET  /live
GET  /health       # backward-compatible liveness alias
GET  /ready
GET  /metadata
POST /predict
POST /predict/batch
```

`MAX_BATCH_SIZE` defaults to 100.

Example single request:

```json
{
  "features": {
    "n_searches": 4,
    "n_nights": 3,
    "filter_price_min": 80,
    "filter_price_max": 180,
    "price_midpoint": 130,
    "price_range": 100,
    "room_type_filter_count": 2,
    "neighborhood_filter_count": 1,
    "has_room_type_filter": 1,
    "has_neighborhood_filter": 1
  }
}
```

The service rejects missing, unexpected, or non-finite features rather than silently reshaping inputs.

## Docker

Batch/training image:

```bash
docker build -t airbnb-demand-proxy-model .
docker run --rm -v "$PWD/outputs:/app/outputs" airbnb-demand-proxy-model
```

API image:

```bash
docker build -f Dockerfile.api -t airbnb-demand-proxy-api .
docker run --rm -p 8000:8000 \
  -v "$PWD/outputs/model_bundle.joblib:/app/models/model_bundle.joblib:ro" \
  airbnb-demand-proxy-api
```

Both deployment paths use non-root container execution.

## Monitoring and governance

The pipeline generates:

- feature PSI and distribution-shift reports
- prediction-vs-target shift summaries
- longitudinal metrics history
- threshold-based monitoring alerts
- model/source/run lineage
- append-only audit events
- run-specific artifact directories
- restricted model/manifest file permissions
- retention metadata

These repository-level controls demonstrate the application layer of MLOps. Production infrastructure would still provide centralized IAM, secrets, telemetry, registry controls, alert routing, orchestration, and rollback execution.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover ingestion, preprocessing, evaluation, training helpers, production artifact creation, and fail-closed data-source behavior.

## Repository structure

```text
.
├── Airbnb Demand Proxy Modeling.ipynb
├── app/
│   └── main.py
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── monitoring.py
│   ├── ops.py
│   ├── preprocess.py
│   ├── run_pipeline.py
│   └── train_model.py
├── tests/
├── docs/
├── Dockerfile
├── Dockerfile.api
└── requirements.txt
```

## Production hardening layer

The API now adds API-key authentication, SHA-256 artifact verification, JSON request logs, request IDs, Prometheus metrics, environment-controlled documentation, CI container builds, Kubernetes probes/autoscaling/disruption controls, and production runbook/security guidance.

The retraining workflow was also changed so a production retrain **cannot silently use synthetic data**. It now requires explicitly staged production data and an immutable source version.

See `docs/production_runbook.md`, `deploy/kubernetes.yaml`, `.env.example`, and `SECURITY.md`.

## Production boundary

Historical outputs in this repository include synthetic development runs. They must not be presented as evidence of real Airbnb demand performance. A live enterprise-production claim still requires deployment on real demand data with real IAM, centralized telemetry, managed registry/artifact delivery, alert routing, canary/rollback execution, outcome feedback, and operational ownership.
