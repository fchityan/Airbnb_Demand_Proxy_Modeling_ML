from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceThresholds:
    mae_max: float = 25.0
    rmse_max: float = 35.0
    r2_min: float = 0.40
    relative_rmse_degradation_max: float = 0.10


def _psi(train_values: np.ndarray, test_values: np.ndarray, bins: int = 10) -> float:
    train_values = train_values[np.isfinite(train_values)]
    test_values = test_values[np.isfinite(test_values)]
    if train_values.size < 2 or test_values.size < 2:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(train_values, quantiles))
    if cut_points.size < 3:
        return 0.0

    train_hist, _ = np.histogram(train_values, bins=cut_points)
    test_hist, _ = np.histogram(test_values, bins=cut_points)

    train_ratio = train_hist / max(train_hist.sum(), 1)
    test_ratio = test_hist / max(test_hist.sum(), 1)

    epsilon = 1e-8
    psi = np.sum((test_ratio - train_ratio) * np.log((test_ratio + epsilon) / (train_ratio + epsilon)))
    return float(psi)


def build_feature_drift_report(x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in x_train.columns:
        train_values = x_train[column].to_numpy(dtype=float)
        test_values = x_test[column].to_numpy(dtype=float)
        rows.append(
            {
                "feature": column,
                "train_mean": float(np.mean(train_values)),
                "test_mean": float(np.mean(test_values)),
                "train_std": float(np.std(train_values)),
                "test_std": float(np.std(test_values)),
                "mean_shift_abs": float(abs(np.mean(test_values) - np.mean(train_values))),
                "psi": _psi(train_values, test_values),
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def build_prediction_shift_report(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    actual_values = y_true.to_numpy(dtype=float)
    rows: list[dict[str, float | str]] = []

    for model_name, y_pred in predictions.items():
        pred_values = np.asarray(y_pred, dtype=float)
        rows.append(
            {
                "model": model_name,
                "actual_mean": float(np.mean(actual_values)),
                "prediction_mean": float(np.mean(pred_values)),
                "actual_std": float(np.std(actual_values)),
                "prediction_std": float(np.std(pred_values)),
                "mean_shift_abs": float(abs(np.mean(pred_values) - np.mean(actual_values))),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_shift_abs", ascending=False).reset_index(drop=True)


def evaluate_metric_alerts(
    metrics_frame: pd.DataFrame,
    thresholds: PerformanceThresholds,
    previous_metrics_for_best_model: pd.DataFrame,
) -> dict[str, float | bool | str]:
    best_row = metrics_frame.sort_values("rmse", ascending=True).iloc[0]
    rmse = float(best_row["rmse"])
    mae = float(best_row["mae"])
    r2 = float(best_row["r2"])
    model_name = str(best_row["model"])

    alerts = {
        "best_model": model_name,
        "mae_alert": mae > thresholds.mae_max,
        "rmse_alert": rmse > thresholds.rmse_max,
        "r2_alert": r2 < thresholds.r2_min,
        "relative_rmse_alert": False,
        "relative_rmse_degradation": 0.0,
    }

    if not previous_metrics_for_best_model.empty:
        previous_best_rmse = float(previous_metrics_for_best_model.iloc[-1]["rmse"])
        if previous_best_rmse > 0:
            degradation = (rmse - previous_best_rmse) / previous_best_rmse
            alerts["relative_rmse_degradation"] = float(degradation)
            alerts["relative_rmse_alert"] = degradation > thresholds.relative_rmse_degradation_max

    return alerts
