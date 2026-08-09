# Model Rollback Plan

## Trigger Conditions
- `monitoring_alerts.json` sets any of `mae_alert`, `rmse_alert`, `r2_alert`, or `relative_rmse_alert` to `true`.
- Online or batch SLO breach for two consecutive monitoring windows.
- Data drift indicator (`psi`) exceeds agreed threshold for critical features.

## Rollback Steps
1. Identify the previous stable run ID from `outputs/metrics_history.csv` and `outputs/runs/<run_id>/run_manifest.json`.
2. Restore the previous model artifact by replacing `outputs/model_bundle.joblib` with the selected `outputs/runs/<run_id>/model_bundle_<run_id>.joblib`.
3. Re-run evaluation using the restored bundle and compare MAE/RMSE/R2 against current production metrics.
4. Update deployment metadata to point to the rolled-back model version and log the operation in `outputs/audit.log`.
5. Open an incident ticket describing the regression root cause and mitigation actions.

## Validation After Rollback
- Confirm performance thresholds are back within limits.
- Confirm latency and throughput SLOs for both batch and online paths.
- Confirm no schema validation failures for current input feed.
