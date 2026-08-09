from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _is_numeric_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def validate_dataframe_schema(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    column_types: dict[str, str] | None = None,
    value_ranges: dict[str, tuple[float, float]] | None = None,
    null_thresholds: dict[str, float] | None = None,
) -> None:
    """Validate schema constraints for production-safe ingestion.

    column_types accepts: "number", "integer", "string".
    null_thresholds represent max null ratio per column (0..1).
    """
    if dataframe.empty:
        raise ValueError("dataframe must not be empty.")

    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}.")

    if column_types:
        for column, expected_type in column_types.items():
            if column not in dataframe.columns:
                continue
            series = dataframe[column]
            if expected_type == "number" and not _is_numeric_dtype(series):
                raise ValueError(f"Column '{column}' must be numeric.")
            if expected_type == "integer" and not pd.api.types.is_integer_dtype(series):
                raise ValueError(f"Column '{column}' must be integer.")
            if expected_type == "string" and not pd.api.types.is_string_dtype(series):
                raise ValueError(f"Column '{column}' must be string.")

    if value_ranges:
        for column, (minimum, maximum) in value_ranges.items():
            if column not in dataframe.columns:
                continue
            series = dataframe[column]
            if not _is_numeric_dtype(series):
                raise ValueError(f"Column '{column}' must be numeric for range validation.")
            in_range_mask = series.between(minimum, maximum, inclusive="both")
            if not in_range_mask.all():
                out_of_range_count = int((~in_range_mask).sum())
                raise ValueError(
                    f"Column '{column}' has {out_of_range_count} rows outside range [{minimum}, {maximum}]."
                )

    if null_thresholds:
        for column, max_ratio in null_thresholds.items():
            if column not in dataframe.columns:
                continue
            if not 0 <= max_ratio <= 1:
                raise ValueError(f"null threshold for '{column}' must be between 0 and 1.")
            null_ratio = float(dataframe[column].isna().mean())
            if np.isnan(null_ratio):
                null_ratio = 1.0
            if null_ratio > max_ratio:
                raise ValueError(
                    f"Column '{column}' null ratio {null_ratio:.4f} exceeds threshold {max_ratio:.4f}."
                )


def split_and_scale(
    dataframe: pd.DataFrame,
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Split the data and scale features for model training."""
    if target_column not in dataframe.columns:
        raise ValueError(f"target_column '{target_column}' is not present in the dataframe.")
    if len(dataframe) < 2:
        raise ValueError("dataframe must contain at least 2 rows.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    if features.empty:
        raise ValueError("dataframe must include at least one feature column.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index,
    )

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler
