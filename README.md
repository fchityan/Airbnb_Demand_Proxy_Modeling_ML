# Airbnb Demand Proxy Modeling ML

## Executive Summary
This repository is a full machine learning pipeline for Airbnb-style demand proxy modeling, covering the complete lifecycle from data ingestion and validation to training, evaluation, monitoring, and model packaging. It combines an exploratory notebook with a modular Python implementation, reproducible execution, automated artifact generation, and practical MLOps workflows.

This project highlights:
- Full end-to-end ML pipeline design with clear module boundaries for ingestion, preprocessing, training, evaluation, and monitoring.
- Run manifests, model versioning, artifact retention, drift tracking, alerting, and rollback planning.
- Baseline and model-based regression benchmarking across multiple algorithms.
- Reproducibility through scripted execution, configuration tracking, versioned outputs, and automated tests.
- Model interpretability artifacts for both linear and tree-based approaches.
- Operational practices for retraining, auditability, and pipeline governance.

## Project Objectives
- Build demand proxy predictions using baseline and ML models across a full data-to-model workflow.
- Compare model quality using MAE, RMSE, and R2 under a repeatable evaluation process.
- Export explainability artifacts for model interpretation and stakeholder review.
- Keep the workflow reproducible, testable, and production-aligned.
- Operationalize the pipeline with MLOps controls for monitoring, versioning, governance, and retraining readiness.

## MLOps Highlights
This project includes practical MLOps patterns that go beyond a single notebook experiment. The pipeline is structured to support day-to-day model operations in a more disciplined way:

- End-to-end lifecycle management: raw data ingestion, schema validation, preprocessing, training, model selection, evaluation, and artifact publication are handled through dedicated modules instead of ad hoc scripts.
- Reproducibility and traceability: each run produces outputs under `outputs/runs/<run_id>/`, a canonical run manifest, and versioned model bundles tied to configuration and dataset metadata.
- Experiment and model governance: model metadata, source lineage, artifact retention policies, and audit logging support accountability across training cycles.
- Model packaging: the trained preprocessor and model are bundled together in a single artifact to reduce train/serve skew and make deployment handoff easier.
- Monitoring and drift detection: feature drift, prediction shift, metric history, and alert thresholds are generated to detect distribution drift and model degradation over time.
- Operational alerting: monitoring artifacts and alert JSON records threshold breaches and relative RMSE degradation so teams can investigate or retrain when needed.
- Retraining and rollback readiness: the repository includes scheduled retraining workflow patterns and a documented rollback plan to support safer model updates.
- CI/CD and quality checks: the project structure anticipates automated validation via tests and workflow automation, which helps maintain reliability before deployment.
- Observability: training and evaluation outputs are stored in a consistent format so dashboards, reports, and incident reviews can use them directly.
- Security and governance controls: restricted file permissions, audit trail capture, and retention metadata help keep the workflow more responsible and transparent.

These MLOps elements make the repository useful both as a learning project and as a template for a more disciplined ML workflow.

## Methodology
- Data preparation: explicit schema validation, train/test split, and standardized numerical features.
- Data sources: the repository includes workbook inputs in the data folder (`contacts.xlsx` and `searches.xlsx`). The loader can read a data directory, merge contacts with searches on the guest/user key, and derive a target from interaction signals such as messages, guests, and searches.
- Models trained: linear regression and XGBoost regressor.
- Prediction approaches evaluated: mean baseline, linear regression, and XGBoost.
- Evaluation metrics: MAE, RMSE, and R2 on holdout data.
- Explainability: linear coefficients and XGBoost feature importances.
- Monitoring: feature drift (PSI), prediction/target distribution shifts, and metric degradation alerts.

## Pipeline Architecture
The workflow is orchestrated through [src/run_pipeline.py](src/run_pipeline.py), which chains together modular components:
- **data_loader**: ingests CSV/Parquet/Excel data and, when given a data directory, reads the included contacts/searches workbooks, joins them, and captures source metadata/versioning.
- **preprocess**: enforces schema checks (required columns, types, ranges, null thresholds), then splits/scales data.
- **train_model**: trains linear regression and XGBoost models.
- **evaluate**: computes metrics, appends metrics history, and emits alert thresholds.
- **monitoring**: writes drift and prediction-shift artifacts for ongoing monitoring.

Run the complete pipeline with `python -m src.run_pipeline` for the default synthetic baseline, or call `run_pipeline()` directly from Python with a data path such as `data/` to process the workbook-based contacts/searches data.

## Project Structure
- `src/` contains the implementation code used by the pipeline.
- `tests/` contains unit tests that verify the behavior of the matching modules in `src/`.
- `outputs/` contains generated artifacts produced by the pipeline.

The similar filenames in `src/` and `tests/` are intentional. For example, `tests/test_preprocess.py` validates the behavior in `src/preprocess.py`. This is a standard layout that keeps module ownership and test coverage easy to follow.

## Purpose of Key Folders
- `src/`
  - Source code for the end-to-end ML workflow.
  - Includes data loading, preprocessing, model training, evaluation, and pipeline orchestration.
  - This is the code you modify when improving model logic or features.
- `tests/`
  - Automated checks for the behavior in `src/`.
  - Helps catch regressions when code changes (for example, incorrect metrics or broken preprocessing).
  - Run before commits/PRs to maintain reliability and reproducibility.
- `outputs/`
  - Run artifacts produced by the pipeline.
  - Intended for analysis/reporting and quick inspection of model quality.
  - Regenerated on each run; treat as derived artifacts, not hand-edited files.

## Outputs
- `outputs/feature_importance.csv`
  - Purpose: interpret which features drive predictions across models.
  - Columns: `model`, `feature`, `importance`, `coefficient`.
  - Notes: `coefficient` is populated for linear regression and `NaN` for XGBoost rows.
- `outputs/validation_metrics.csv`
  - Purpose: compare model quality on holdout data.
  - Columns: `model`, `mae`, `rmse`, `r2`.
  - Notes: rows are sorted by ascending `rmse`, so the first row is the top performer by RMSE.
- `outputs/summary.json`
  - Purpose: provide a quick, machine-readable report for downstream use.
  - Includes: evaluated models list, best model by RMSE, actual-target summary statistics, and per-model prediction summaries.
- `outputs/run_manifest.json`
  - Purpose: canonical run metadata tying data source, training config, model version, artifacts, SLOs, and governance settings to one run ID.
- `outputs/model_bundle.joblib`
  - Purpose: export preprocessing + selected model together to avoid train/serve skew.
- `outputs/runs/<run_id>/...`
  - Purpose: immutable versioned artifact store per training run.
- `outputs/metrics_history.csv`
  - Purpose: longitudinal MAE/RMSE/R2 tracking across runs.
- `outputs/drift_report.csv` and `outputs/prediction_shift.csv`
  - Purpose: monitor feature drift and target/prediction distribution shifts.
- `outputs/monitoring_alerts.json`
  - Purpose: alerts for threshold breaches and relative RMSE degradation.
- `outputs/audit.log`
  - Purpose: append-only audit trail of run lifecycle events.

## Quick Run
- Install dependencies:
  - `pip install -r requirements.txt`
- Run the default pipeline:
  - `python -m src.run_pipeline`
- Run the workbook-based workflow with the included data folder:
  - `python -c "from pathlib import Path; from src.run_pipeline import run_pipeline; run_pipeline(output_dir=Path('outputs'), data_path='data')"`
- Run tests:
  - `python -m unittest discover -s tests -v`

## Validation and Error Handling
- `load_data` validates that `n_samples >= 2`.
- `split_and_scale` validates target-column existence, non-empty features, minimum row count, and `test_size` bounds.
- `evaluate_models_and_save_outputs` validates that predictions are non-empty, aligned to ground-truth length, and finite.

## Deployment Considerations
- Data and schema management
  - Production ingestion now supports CSV/Parquet and captures source metadata (source version, row count, timestamp, dataset fingerprint).
  - Schema checks are enforced via `validate_dataframe_schema` for required columns, data types, ranges, and null thresholds.
- Reproducibility and model versioning
  - Dependencies are pinned in `requirements.txt`.
  - Each run records full training configuration and emits a versioned `model_bundle_<run_id>.joblib`.
  - All outputs are tied to run-specific directories under `outputs/runs/<run_id>/`.
- Serving and inference
  - Preprocessor + trained model are exported as one bundle.
  - Batch/online latency and throughput SLOs are captured in the run manifest.
- Monitoring and drift
  - Drift artifacts include PSI-based feature drift and prediction/target distribution shift reports.
  - Metric history and alert thresholds are generated per run for MAE/RMSE/R2 degradation tracking.
- Security and governance
  - No raw records are persisted in artifacts; only aggregate metadata/statistics are stored.
  - Model bundles and manifests are written with restricted file permissions.
  - Audit logging and retention policy metadata are included.
- CI/CD and operational readiness
  - PR/unit-test workflow in `.github/workflows/ci.yml`.
  - Scheduled retraining workflow in `.github/workflows/retrain.yml`.
  - Rollback procedure in `docs/rollback_plan.md`.

## Limitations
- Synthetic fallback still exists for local/testing scaffolding when no input path is provided.
- Access control enforcement in this repository is file-permission based; enterprise IAM/RBAC integration must be configured in deployment infrastructure.
- Additional model comparison and hyperparameter optimization can further improve robustness.
