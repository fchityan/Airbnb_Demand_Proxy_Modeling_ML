from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.evaluate import calculate_regression_metrics
from src.monitoring import (
    PerformanceThresholds,
    build_feature_drift_report,
    build_prediction_shift_report,
    evaluate_metric_alerts,
)
from src.preprocess import validate_dataframe_schema
from src.train_model import build_feature_importance_for_models, predict_mean_baseline, train_models


def _set_restricted_permissions(path: Path) -> None:
    path.chmod(0o600)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_audit_event(output_dir: Path, event: dict[str, str]) -> None:
    with (output_dir / "audit.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scale_split(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X_other), columns=X_train.columns, index=X_other.index)


def run_pipeline(
    output_dir: Path,
    data_path: str | Path | None = None,
    source_name: str = "production_source",
    source_version: str = "v1",
    random_state: int = 42,
    n_samples: int = 2000,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    run_id: str | None = None,
) -> dict[str, str]:
    """Train, validate, lock a model, evaluate on holdout data, and package artifacts."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5.")
    if not 0 < validation_size < 0.5:
        raise ValueError("validation_size must be between 0 and 0.5.")
    if test_size + validation_size >= 0.8:
        raise ValueError("test_size + validation_size leaves too little training data.")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    resolved_run_id = run_id or f"run_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    run_dir = output_dir / "runs" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    thresholds = PerformanceThresholds()

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

    required_columns = list(dataframe.columns)
    validate_dataframe_schema(
        dataframe,
        required_columns=required_columns,
        column_types={column: "number" for column in required_columns},
        null_thresholds={column: 0.05 for column in required_columns},
    )

    features = dataframe.drop(columns=["target"])
    target = dataframe["target"]
    if features.empty:
        raise ValueError("At least one feature column is required.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=test_size + validation_size,
        random_state=random_state,
    )
    relative_test_size = test_size / (test_size + validation_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
    )

    selection_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        selection_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = _scale_split(X_train, X_val, selection_scaler)

    selection_models = train_models(X_train_scaled, y_train, random_state=random_state)
    validation_predictions = {
        "mean_baseline": predict_mean_baseline(y_train, len(X_val_scaled)),
        "linear_regression": selection_models["linear_regression"].predict(X_val_scaled),
        "xgboost": selection_models["xgboost"].predict(X_val_scaled),
    }

    validation_rows = []
    for model_name, predictions in validation_predictions.items():
        validation_rows.append({"model": model_name, **calculate_regression_metrics(y_val, predictions)})
    validation_metrics = pd.DataFrame(validation_rows).sort_values("rmse").reset_index(drop=True)
    validation_metrics.to_csv(run_dir / "validation_model_comparison.csv", index=False)

    baseline_rmse = float(
        validation_metrics.loc[validation_metrics["model"] == "mean_baseline", "rmse"].iloc[0]
    )
    ml_candidates = validation_metrics[validation_metrics["model"].isin(["linear_regression", "xgboost"])]
    best_ml_row = ml_candidates.sort_values("rmse").iloc[0]
    best_model_name = str(best_ml_row["model"])
    best_validation_rmse = float(best_ml_row["rmse"])
    approved_for_production = best_validation_rmse < baseline_rmse

    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    final_scaler = StandardScaler()
    X_trainval_scaled = pd.DataFrame(
        final_scaler.fit_transform(X_trainval), columns=X_trainval.columns, index=X_trainval.index
    )
    X_test_scaled = pd.DataFrame(
        final_scaler.transform(X_test), columns=X_trainval.columns, index=X_test.index
    )
    final_models = train_models(X_trainval_scaled, y_trainval, random_state=random_state)
    final_model = final_models[best_model_name]
    test_prediction = final_model.predict(X_test_scaled)
    test_metrics = calculate_regression_metrics(y_test, test_prediction)
    baseline_test_prediction = predict_mean_baseline(y_trainval, len(X_test_scaled))
    baseline_test_metrics = calculate_regression_metrics(y_test, baseline_test_prediction)

    test_metrics_frame = pd.DataFrame(
        [
            {"model": best_model_name, **test_metrics},
            {"model": "mean_baseline", **baseline_test_metrics},
        ]
    ).sort_values("rmse")
    test_metrics_frame.to_csv(run_dir / "test_metrics.csv", index=False)

    feature_importance = build_feature_importance_for_models(
        {best_model_name: final_model}, list(X_trainval_scaled.columns)
    )
    feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)

    drift_frame = build_feature_drift_report(X_trainval_scaled, X_test_scaled)
    drift_frame.to_csv(run_dir / "drift_report.csv", index=False)
    prediction_shift = build_prediction_shift_report(
        y_test,
        {best_model_name: test_prediction, "mean_baseline": baseline_test_prediction},
    )
    prediction_shift.to_csv(run_dir / "prediction_shift.csv", index=False)

    model_version = f"{best_model_name}_{resolved_run_id}"
    training_config = {
        "random_state": random_state,
        "test_size": test_size,
        "validation_size": validation_size,
        "n_samples": n_samples,
        "target_column": "target",
        "target_semantics": "guest_count for workbook-based production input",
        "feature_columns": list(X_trainval.columns),
        "selection_metric": "validation_rmse",
        "baseline_gate": "selected ML validation RMSE must be lower than mean baseline RMSE",
    }
    bundle = {
        "run_id": resolved_run_id,
        "model_version": model_version,
        "model_name": best_model_name,
        "preprocessor": final_scaler,
        "model": final_model,
        "feature_columns": list(X_trainval.columns),
        "training_config": training_config,
        "source_metadata": source_metadata,
        "promotion_status": "approved" if approved_for_production else "rejected_baseline_gate",
        "validation_metrics": best_ml_row.to_dict(),
        "test_metrics": test_metrics,
    }
    versioned_model_path = run_dir / f"model_bundle_{resolved_run_id}.joblib"
    latest_model_path = output_dir / "model_bundle.joblib"
    joblib.dump(bundle, versioned_model_path)
    shutil.copy2(versioned_model_path, latest_model_path)
    _set_restricted_permissions(versioned_model_path)
    _set_restricted_permissions(latest_model_path)
    model_sha256 = _sha256_file(versioned_model_path)

    summary = {
        "run_id": resolved_run_id,
        "best_model_by_validation_rmse": best_model_name,
        "promotion_status": bundle["promotion_status"],
        "validation_baseline_rmse": baseline_rmse,
        "validation_model_rmse": best_validation_rmse,
        "test_metrics": test_metrics,
        "test_baseline_metrics": baseline_test_metrics,
        "target_semantics": training_config["target_semantics"],
    }
    _write_json(run_dir / "summary.json", summary)

    history_path = output_dir / "metrics_history.csv"
    history_row = pd.DataFrame(
        [
            {
                "run_id": resolved_run_id,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": best_model_name,
                **test_metrics,
            }
        ]
    )
    previous_history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    pd.concat([previous_history, history_row], ignore_index=True).to_csv(history_path, index=False)
    previous_same_model = (
        previous_history[previous_history["model"] == best_model_name]
        if not previous_history.empty and "model" in previous_history
        else pd.DataFrame()
    )
    alerts = evaluate_metric_alerts(history_row, thresholds, previous_same_model)
    _write_json(
        run_dir / "monitoring_alerts.json",
        {
            "run_id": resolved_run_id,
            "promotion_status": bundle["promotion_status"],
            "alerts": alerts,
        },
    )

    run_manifest = {
        "run_id": resolved_run_id,
        "run_timestamp_utc": timestamp.isoformat(),
        "source_metadata": source_metadata,
        "training_config": training_config,
        "model_version": model_version,
        "model_sha256": model_sha256,
        "promotion_status": bundle["promotion_status"],
        "artifact_paths": {
            "run_dir": str(run_dir),
            "validation_metrics": str(run_dir / "validation_model_comparison.csv"),
            "test_metrics": str(run_dir / "test_metrics.csv"),
            "summary": str(run_dir / "summary.json"),
            "drift_report": str(run_dir / "drift_report.csv"),
            "prediction_shift": str(run_dir / "prediction_shift.csv"),
            "model_bundle": str(latest_model_path),
            "model_bundle_versioned": str(versioned_model_path),
        },
        "slo": {
            "batch": {"p95_latency_ms_max": 3000, "throughput_rows_per_sec_min": 500},
            "online": {"p95_latency_ms_max": 120, "throughput_rps_min": 30},
        },
        "governance": {
            "raw_data_persisted": False,
            "retention_days": 30,
            "artifact_access_mode": "owner_read_write",
        },
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "run_manifest.json", run_manifest)
    _set_restricted_permissions(run_dir / "run_manifest.json")
    _set_restricted_permissions(output_dir / "run_manifest.json")

    for source, target_path in [
        (run_dir / "feature_importance.csv", output_dir / "feature_importance.csv"),
        (run_dir / "validation_model_comparison.csv", output_dir / "validation_metrics.csv"),
        (run_dir / "test_metrics.csv", output_dir / "test_metrics.csv"),
        (run_dir / "summary.json", output_dir / "summary.json"),
        (run_dir / "monitoring_alerts.json", output_dir / "monitoring_alerts.json"),
        (run_dir / "drift_report.csv", output_dir / "drift_report.csv"),
        (run_dir / "prediction_shift.csv", output_dir / "prediction_shift.csv"),
    ]:
        shutil.copy2(source, target_path)

    _append_audit_event(
        output_dir,
        {
            "event": "pipeline_run_completed",
            "run_id": resolved_run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
            "promotion_status": bundle["promotion_status"],
        },
    )

    return {
        "run_id": resolved_run_id,
        "summary": str(output_dir / "summary.json"),
        "run_manifest": str(output_dir / "run_manifest.json"),
        "model_bundle": str(latest_model_path),
    }


if __name__ == "__main__":
    generated = run_pipeline(output_dir=Path("outputs"))
    for name, path in generated.items():
        print(f"{name}: {path}")
