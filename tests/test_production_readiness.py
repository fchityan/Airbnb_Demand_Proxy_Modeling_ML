from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.preprocess import validate_dataframe_schema
from src.run_pipeline import run_pipeline


class ProductionReadinessTests(unittest.TestCase):
    def test_load_data_from_csv_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "input.csv"
            pd.DataFrame(
                {
                    "feature_1": [1.0, 2.0, 3.0],
                    "feature_2": [4.0, 5.0, 6.0],
                    "target": [10.0, 11.0, 12.0],
                }
            ).to_csv(csv_path, index=False)

            dataframe, metadata = load_data(
                data_path=csv_path,
                source_name="local_csv",
                source_version="v1",
                return_metadata=True,
            )

            self.assertEqual(len(dataframe), 3)
            self.assertEqual(metadata["source_name"], "local_csv")
            self.assertEqual(metadata["row_count"], 3)

    def test_validate_dataframe_schema_rejects_invalid_ranges(self) -> None:
        dataframe = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0],
                "feature_2": [4.0, 5.0, 6.0],
                "target": [100.0, 2.0, 3.0],
            }
        )

        with self.assertRaises(ValueError):
            validate_dataframe_schema(
                dataframe,
                required_columns=["feature_1", "feature_2", "target"],
                value_ranges={"target": (0.0, 50.0)},
            )

    def test_run_pipeline_writes_manifest_and_model_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_pipeline(output_dir=output_dir, n_samples=80, test_size=0.2)

            self.assertIn("summary", result)
            self.assertTrue((output_dir / "run_manifest.json").exists())
            self.assertTrue((output_dir / "model_bundle.joblib").exists())


if __name__ == "__main__":
    unittest.main()
