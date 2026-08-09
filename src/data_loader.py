from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression


def _load_excel_workbook(directory_path: Path) -> pd.DataFrame:
    contacts_path = directory_path / "contacts.xlsx"
    searches_path = directory_path / "searches.xlsx"
    if not contacts_path.exists() or not searches_path.exists():
        raise FileNotFoundError("Expected contacts.xlsx and searches.xlsx in the provided data directory.")

    contacts = pd.read_excel(contacts_path, sheet_name="contacts")
    searches = pd.read_excel(searches_path, sheet_name="searches")

    merged = contacts.merge(searches, left_on="id_guest", right_on="id_user", how="left")

    merged = merged.copy()
    merged["n_searches"] = merged["n_searches"].fillna(0).astype(int)
    merged["n_messages"] = merged["n_messages"].fillna(0).astype(int)
    merged["n_guests"] = merged["n_guests"].fillna(0).astype(int)
    merged["n_nights"] = merged["n_nights"].fillna(0).astype(float)
    merged["filter_price_min"] = merged["filter_price_min"].fillna(0.0)
    merged["filter_price_max"] = merged["filter_price_max"].fillna(0.0)
    merged["filter_room_types"] = merged["filter_room_types"].fillna("")
    merged["filter_neighborhoods"] = merged["filter_neighborhoods"].fillna("")
    merged["origin_country"] = merged["origin_country"].fillna("UNK")

    feature_columns = [
        "n_messages",
        "n_guests",
        "n_searches",
        "n_nights",
        "filter_price_min",
        "filter_price_max",
        "origin_country",
        "filter_room_types",
        "filter_neighborhoods",
    ]
    for column in feature_columns:
        if column not in merged.columns:
            merged[column] = 0

    merged = merged.dropna(subset=["id_guest", "id_host", "id_listing"])
    if merged.empty:
        raise ValueError("Merged contact/search data is empty after cleaning.")

    return merged


def _fingerprint_dataframe(dataframe: pd.DataFrame) -> str:
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()[:16]


def load_data(
    data_path: str | Path | None = None,
    random_state: int = 42,
    n_samples: int = 2000,
    source_name: str = "synthetic_generator",
    source_version: str = "dev",
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, str | int]]:
    """Load production data from disk, with synthetic fallback for local development."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    path = Path(data_path) if data_path is not None else None
    if path is not None:
        if not path.exists():
            features, target = make_regression(
                n_samples=n_samples,
                n_features=12,
                n_informative=8,
                n_targets=1,
                noise=12.0,
                random_state=random_state,
            )
            columns = [f"feature_{index}" for index in range(features.shape[1])]
            dataframe = pd.DataFrame(features, columns=columns)
            dataframe["target"] = target
            source_name_resolved = "synthetic_generator"
        elif path.is_dir():
            try:
                dataframe = _load_excel_workbook(path)
            except FileNotFoundError:
                features, target = make_regression(
                    n_samples=n_samples,
                    n_features=12,
                    n_informative=8,
                    n_targets=1,
                    noise=12.0,
                    random_state=random_state,
                )
                columns = [f"feature_{index}" for index in range(features.shape[1])]
                dataframe = pd.DataFrame(features, columns=columns)
                dataframe["target"] = target
                source_name_resolved = "synthetic_generator"
            else:
                source_name_resolved = source_name
        elif path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(path)
            source_name_resolved = source_name
        elif path.suffix.lower() in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(path)
            source_name_resolved = source_name
        elif path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            dataframe = pd.read_excel(path)
            source_name_resolved = source_name
        else:
            raise ValueError("Unsupported data format. Use .csv, .parquet, or Excel files.")
    else:
        features, target = make_regression(
            n_samples=n_samples,
            n_features=12,
            n_informative=8,
            n_targets=1,
            noise=12.0,
            random_state=random_state,
        )

        columns = [f"feature_{index}" for index in range(features.shape[1])]
        dataframe = pd.DataFrame(features, columns=columns)
        dataframe["target"] = target
        source_name_resolved = "synthetic_generator"

    if "target" not in dataframe.columns:
        dataframe = dataframe.copy()
        dataframe["target"] = dataframe["n_messages"] + dataframe["n_searches"] + dataframe["n_guests"]

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
