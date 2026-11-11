# AI Readiness & Model Governance Assessment — Airbnb Guest Demand Proxy

## Data readiness

The regression target is `n_guests`. Search-frequency, stay-length and filter features are used to predict the target; identifiers and post-outcome information must be excluded. Real workbook data and model-ready tabular inputs are supported.

**Evidence boundary:** Historical outputs include synthetic development runs. They demonstrate pipeline behaviour, **not validated real Airbnb demand prediction**.

## Model lifecycle and governance

| Capability | Existing implementation | Required before production adoption |
|---|---|---|
| Data quality | Schema and source validation; no silent real-data → synthetic fallback | Source-owner approval, quality thresholds, access controls |
| Baseline and modelling | Mean predictor, Linear Regression, XGBoost | Independent real-data benchmark and business baseline |
| Evaluation | Validation RMSE with held-out test | Population representativeness and drift monitoring validation |
| Model lineage | Versioned bundle, source/run metadata and checksums | Release approval and model inventory ownership |
| Monitoring | Prediction/feature-shift reports and alerts | Assigned alert-response process and thresholds |
| Business value | Guest count proxy | Pilot demonstrating benefit against an agreed demand-planning KPI |

**AI-enablement assessment:** a reproducible reference implementation requiring an independently evaluated real-data pilot before live business use.

See [requirements and solution design](../business-analysis/business-requirements-and-solution-design.md).
