from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib

from src.data_loader import load_data
from src.evaluate import evaluate_models_and_save_outputs
from src.monitoring import (
    PerformanceThresholds,
    build_feature_drift_report,
    build_prediction_shift_report,
)
from src.preprocess import split_and_scale, validate_dataframe_schema
from src.train_model import (
    build_feature_importance_for_models,
    predict_mean_baseline,
    train_models,
)


def _set_restricted_permissions(path: Path) -> None:
    path.chmod(0o600)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def _append_audit_event(output_dir: Path, event: dict[str, str]) -> None:
    audit_path = output_dir / "audit.log"
    with audit_path.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(event) + "\n")


def run_pipeline(
    output_dir: Path,
    data_path: str | Path | None = None,
    source_name: str = "production_source",
    source_version: str = "v1",
    random_state: int = 42,
    n_samples: int = 2000,
    test_size: float = 0.2,
    run_id: str | None = None,
) -> dict[str, str]:
    """Train models, evaluate them, and write versioned production artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    resolved_run_id = run_id or f"run_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    run_dir = output_dir / "runs" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    performance_thresholds = PerformanceThresholds()

    _append_audit_event(
        output_dir,
        {
            "event": "pipeline_run_started",
            "run_id": resolved_run_id,
            "timestamp_utc": timestamp.isoformat(),
            "source_version": source_version,
        },
    )

    dataframe, source_metadata = load_data(
        data_path=data_path,
        random_state=random_state,
        n_samples=n_samples,
        source_name=source_name,
        source_version=source_version,
        return_metadata=True,
    )

    required_columns = ["target"] + [column for column in dataframe.columns if column != "target"]
    numeric_columns = {column: "number" for column in required_columns}
    null_thresholds = {column: 0.05 for column in required_columns}
    validate_dataframe_schema(
        dataframe,
        required_columns=required_columns,
        column_types=numeric_columns,
        value_ranges={},
        null_thresholds=null_thresholds,
    )

    x_train, x_test, y_train, y_test, scaler = split_and_scale(
        dataframe,
        test_size=test_size,
        random_state=random_state,
    )
    models = train_models(x_train, y_train, random_state=random_state)

    predictions = {
        "mean_baseline": predict_mean_baseline(y_train, len(x_test)),
        "linear_regression": models["linear_regression"].predict(x_test),
        "xgboost": models["xgboost"].predict(x_test),
    }

    feature_importance = build_feature_importance_for_models(models, list(x_train.columns))
    feature_importance_path = run_dir / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    metrics_frame = evaluate_models_and_save_outputs(
        y_true=y_test,
        predictions=predictions,
        output_dir=run_dir,
        run_id=resolved_run_id,
        thresholds=performance_thresholds,
    )

    drift_frame = build_feature_drift_report(x_train, x_test)
    drift_path = run_dir / "drift_report.csv"
    drift_frame.to_csv(drift_path, index=False)

    prediction_shift_frame = build_prediction_shift_report(y_test, predictions)
    prediction_shift_path = run_dir / "prediction_shift.csv"
    prediction_shift_frame.to_csv(prediction_shift_path, index=False)

    best_model_name = str(metrics_frame.sort_values("rmse", ascending=True).iloc[0]["model"])
    model_to_serve = models[best_model_name]
    model_version_tag = f"{best_model_name}_{resolved_run_id}"

    training_config = {
        "random_state": random_state,
        "test_size": test_size,
        "n_samples": n_samples,
        "target_column": "target",
        "feature_columns": list(x_train.columns),
    }

    model_bundle_payload = {
        "run_id": resolved_run_id,
        "model_version": model_version_tag,
        "model_name": best_model_name,
        "preprocessor": scaler,
        "model": model_to_serve,
        "feature_columns": list(x_train.columns),
        "training_config": training_config,
        "source_metadata": source_metadata,
    }
    model_bundle_versioned_path = run_dir / f"model_bundle_{resolved_run_id}.joblib"
    model_bundle_latest_path = output_dir / "model_bundle.joblib"
    joblib.dump(model_bundle_payload, model_bundle_versioned_path)
    shutil.copy2(model_bundle_versioned_path, model_bundle_latest_path)
    _set_restricted_permissions(model_bundle_versioned_path)
    _set_restricted_permissions(model_bundle_latest_path)

    run_manifest = {
        "run_id": resolved_run_id,
        "run_timestamp_utc": timestamp.isoformat(),
        "source_metadata": source_metadata,
        "training_config": training_config,
        "model_version": model_version_tag,
        "artifact_paths": {
            "run_dir": str(run_dir),
            "feature_importance": str(feature_importance_path),
            "validation_metrics": str(run_dir / "validation_metrics.csv"),
            "summary": str(run_dir / "summary.json"),
            "drift_report": str(drift_path),
            "prediction_shift": str(prediction_shift_path),
            "model_bundle": str(model_bundle_latest_path),
            "model_bundle_versioned": str(model_bundle_versioned_path),
        },
        "slo": {
            "batch": {"p95_latency_ms_max": 3000, "throughput_rows_per_sec_min": 500},
            "online": {"p95_latency_ms_max": 120, "throughput_rps_min": 30},
        },
        "monitoring_thresholds": {
            "mae_max": performance_thresholds.mae_max,
            "rmse_max": performance_thresholds.rmse_max,
            "r2_min": performance_thresholds.r2_min,
            "relative_rmse_degradation_max": performance_thresholds.relative_rmse_degradation_max,
        },
        "governance": {
            "raw_data_persisted": False,
            "audit_log": str(output_dir / "audit.log"),
            "retention_days": 30,
            "artifact_access_mode": "owner_read_write",
        },
    }

    run_manifest_versioned_path = run_dir / "run_manifest.json"
    run_manifest_latest_path = output_dir / "run_manifest.json"
    _write_json(run_manifest_versioned_path, run_manifest)
    _write_json(run_manifest_latest_path, run_manifest)
    _set_restricted_permissions(run_manifest_versioned_path)
    _set_restricted_permissions(run_manifest_latest_path)

    retention_policy_path = output_dir / "RETENTION_POLICY.json"
    if not retention_policy_path.exists():
        _write_json(
            retention_policy_path,
            {
                "retention_days": 30,
                "delete_strategy": "runs_older_than_retention",
                "audit_log_required": True,
            },
        )

    compatibility_files = [
        (run_dir / "feature_importance.csv", output_dir / "feature_importance.csv"),
        (run_dir / "validation_metrics.csv", output_dir / "validation_metrics.csv"),
        (run_dir / "summary.json", output_dir / "summary.json"),
        (run_dir / "monitoring_alerts.json", output_dir / "monitoring_alerts.json"),
    ]
    for source_path, target_path in compatibility_files:
        if source_path.exists():
            shutil.copy2(source_path, target_path)

    _append_audit_event(
        output_dir,
        {
            "event": "pipeline_run_completed",
            "run_id": resolved_run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version_tag,
        },
    )

    return {
        "run_id": resolved_run_id,
        "feature_importance": str(feature_importance_path),
        "validation_metrics": str(run_dir / "validation_metrics.csv"),
        "summary": str(run_dir / "summary.json"),
        "run_manifest": str(run_manifest_latest_path),
        "model_bundle": str(model_bundle_latest_path),
    }


if __name__ == "__main__":
    generated = run_pipeline(output_dir=Path("outputs"))
    for name, path in generated.items():
        print(f"{name}: {path}")