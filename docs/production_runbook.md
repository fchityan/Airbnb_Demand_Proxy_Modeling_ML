# Production Runbook

The repository now models the application/MLOps controls of a production regression service. A live-production claim requires real demand data, deployed infrastructure, and an operating team.

## Release and promotion

1. Stage the approved data source; production training is fail-closed and must never fall back to synthetic data.
2. Run the pipeline with an immutable source version.
3. Review validation model comparison, baseline promotion gate, locked test metrics, drift, prediction shift, alerts, source lineage, and the run manifest.
4. Promote only a run whose promotion status is approved and whose model fingerprint is recorded.
5. Store the API key and promoted model SHA-256 in the deployment secret manager.
6. Deploy the immutable container image plus immutable model artifact.
7. Verify `/live`, `/ready`, `/metrics`, metadata, and a known-good prediction before routing traffic.

The GitHub retraining workflow no longer schedules synthetic retraining. It is a fail-closed template that requires explicitly staged production data.

## Runtime controls

- `/live` and `/health`: liveness.
- `/ready`: confirms that the model is loaded and verified.
- `/metrics`: Prometheus request count, latency, and model-info metrics.
- `/metadata`: authenticated lineage, promotion status, schema, and model fingerprint.
- `/predict` and `/predict/batch`: authenticated inference with batch bounds.
- Every response receives an `X-Request-ID`.
- Request logs are JSON and exclude feature payloads.
- `MODEL_SHA256` pins the exact promoted artifact.
- `ENABLE_DOCS=false` hides Swagger/OpenAPI endpoints in production.

## Suggested SLO targets

Targets only; validate them in the real environment:

- 99.9% monthly inference availability.
- p95 single-record service latency below 150 ms.
- 5xx rate below 0.5% over 15 minutes.
- no routing to replicas failing model readiness.

The run manifest already contains example online/batch engineering targets; production dashboards should use measurements from actual traffic.

## Monitoring and model risk

Join service telemetry with feature drift, prediction shift, delayed outcome labels, baseline comparisons, and business metrics. A model that no longer beats the baseline should not be automatically promoted.

## Rollback

Use immutable model/run IDs and image digests. Roll back the image and model together to the prior verified release. Keep the previous run manifest and checksum for audit.

## Still external to this repository

Real production still requires cloud IAM, managed secrets, centralized logs/traces, an artifact/model registry, object-store/data-pipeline integration, gateway rate limiting/WAF, image signing/scanning, alert routing, canary rollout, real outcome feedback, and on-call ownership.
