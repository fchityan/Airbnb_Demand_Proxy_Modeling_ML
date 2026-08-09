from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression


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
            raise FileNotFoundError(f"Input data file not found: {path}")
        if path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported data format. Use .csv or .parquet.")
        source_name_resolved = source_name
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
