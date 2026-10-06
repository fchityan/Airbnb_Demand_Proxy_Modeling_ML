from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression

MODEL_FEATURES = [
    "n_searches",
    "n_nights",
    "filter_price_min",
    "filter_price_max",
    "price_midpoint",
    "price_range",
    "room_type_filter_count",
    "neighborhood_filter_count",
    "has_room_type_filter",
    "has_neighborhood_filter",
]
TARGET_COLUMN = "target"


def _selection_count(value: object) -> int:
    """Estimate the number of selections stored in a workbook filter field."""
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    for separator in ["|", ",", ";"]:
        if separator in text:
            return len([part for part in text.split(separator) if part.strip()])
    return 1


def _load_excel_workbook(directory_path: Path) -> pd.DataFrame:
    """Build the production model frame from contact/search workbooks.

    The original notebook selected guest count as the final regression target. The
    production frame therefore predicts n_guests and explicitly removes n_guests,
    identifiers, and post-contact message counts from model features.
    """
    contacts_path = directory_path / "contacts.xlsx"
    searches_path = directory_path / "searches.xlsx"
    if not contacts_path.exists() or not searches_path.exists():
        raise FileNotFoundError("Expected contacts.xlsx and searches.xlsx in the provided data directory.")

    contacts = pd.read_excel(contacts_path, sheet_name="contacts")
    searches = pd.read_excel(searches_path, sheet_name="searches")
    merged = contacts.merge(searches, left_on="id_guest", right_on="id_user", how="left")
    merged = merged.copy()

    required = ["n_guests", "n_searches", "n_nights", "filter_price_min", "filter_price_max"]
    missing = [column for column in required if column not in merged.columns]
    if missing:
        raise ValueError(f"Workbook data is missing required modeling columns: {missing}")

    for column in ["n_guests", "n_searches", "n_nights", "filter_price_min", "filter_price_max"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    merged["n_searches"] = merged["n_searches"].fillna(0.0)
    merged["n_nights"] = merged["n_nights"].fillna(0.0)
    merged["filter_price_min"] = merged["filter_price_min"].fillna(0.0)
    merged["filter_price_max"] = merged["filter_price_max"].fillna(0.0)

    room_types = merged.get("filter_room_types", pd.Series("", index=merged.index))
    neighborhoods = merged.get("filter_neighborhoods", pd.Series("", index=merged.index))
    merged["room_type_filter_count"] = room_types.map(_selection_count).astype(float)
    merged["neighborhood_filter_count"] = neighborhoods.map(_selection_count).astype(float)
    merged["has_room_type_filter"] = (merged["room_type_filter_count"] > 0).astype(float)
    merged["has_neighborhood_filter"] = (merged["neighborhood_filter_count"] > 0).astype(float)
    merged["price_midpoint"] = (merged["filter_price_min"] + merged["filter_price_max"]) / 2.0
    merged["price_range"] = (merged["filter_price_max"] - merged["filter_price_min"]).clip(lower=0.0)

    frame = merged[MODEL_FEATURES + ["n_guests"]].rename(columns={"n_guests": TARGET_COLUMN})
    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].clip(lower=0.0)
    if frame.empty:
        raise ValueError("No rows with a usable guest-count target remain after cleaning.")
    return frame


def _fingerprint_dataframe(dataframe: pd.DataFrame) -> str:
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()[:16]


def _synthetic_frame(n_samples: int, random_state: int) -> pd.DataFrame:
    features, target = make_regression(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_targets=1,
        noise=12.0,
        random_state=random_state,
    )
    columns = [f"feature_{index}" for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=columns)
    frame[TARGET_COLUMN] = target
    return frame


def _normalize_external_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    if TARGET_COLUMN not in frame.columns:
        if "n_guests" in frame.columns:
            frame = frame.rename(columns={"n_guests": TARGET_COLUMN})
        else:
            raise ValueError("Input data must include 'target' or the notebook-aligned 'n_guests' target column.")
    if "n_guests" in frame.columns and "n_guests" != TARGET_COLUMN:
        frame = frame.drop(columns=["n_guests"])
    non_numeric = [
        column
        for column in frame.columns
        if column != TARGET_COLUMN and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            "Production CSV/Parquet inputs must be model-ready numeric features. "
            f"Non-numeric columns found: {non_numeric}. Use the workbook directory loader for raw Airbnb-style inputs."
        )
    return frame


def load_data(
    data_path: str | Path | None = None,
    random_state: int = 42,
    n_samples: int = 2000,
    source_name: str = "synthetic_generator",
    source_version: str = "dev",
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, str | int]]:
    """Load model data with explicit failure for missing production inputs.

    Synthetic data is generated only when no data_path is supplied. A provided but
    missing/invalid path fails closed instead of silently training a synthetic model.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    path = Path(data_path) if data_path is not None else None
    if path is None:
        dataframe = _synthetic_frame(n_samples, random_state)
        source_name_resolved = "synthetic_generator"
    else:
        if not path.exists():
            raise FileNotFoundError(f"Configured data source does not exist: {path}")
        if path.is_dir():
            dataframe = _load_excel_workbook(path)
        elif path.suffix.lower() == ".csv":
            dataframe = _normalize_external_frame(pd.read_csv(path))
        elif path.suffix.lower() in {".parquet", ".pq"}:
            dataframe = _normalize_external_frame(pd.read_parquet(path))
        elif path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            dataframe = _normalize_external_frame(pd.read_excel(path))
        else:
            raise ValueError("Unsupported data format. Use .csv, .parquet, Excel, or the workbook directory.")
        source_name_resolved = source_name

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Model frame must include '{TARGET_COLUMN}'.")

    metadata: dict[str, str | int] = {
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name_resolved,
        "source_version": source_version,
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "dataset_fingerprint": _fingerprint_dataframe(dataframe),
    }
    if path is not None:
        metadata["source_path"] = str(path)

    if return_metadata:
        return dataframe, metadata
    return dataframe
