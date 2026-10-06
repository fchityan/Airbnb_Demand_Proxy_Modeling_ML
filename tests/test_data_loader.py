from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_loader import MODEL_FEATURES, load_data


class DataLoaderTests(unittest.TestCase):
    def test_load_data_has_target_column_for_explicit_dev_synthetic_mode(self) -> None:
        dataframe = load_data(n_samples=20, random_state=42)

        self.assertIn("target", dataframe.columns)
        self.assertEqual(len(dataframe), 20)

    def test_load_data_fails_closed_when_configured_path_is_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_data(data_path="definitely_missing_data_dir", n_samples=20, random_state=42)

    def test_external_guest_target_is_removed_from_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "model_ready.csv"
            payload = {feature: [1.0, 2.0, 3.0] for feature in MODEL_FEATURES}
            payload["n_guests"] = [1.0, 2.0, 4.0]
            pd.DataFrame(payload).to_csv(csv_path, index=False)

            dataframe = load_data(data_path=csv_path)

            self.assertIn("target", dataframe.columns)
            self.assertNotIn("n_guests", dataframe.columns)
            self.assertEqual(dataframe["target"].tolist(), [1.0, 2.0, 4.0])

    def test_load_data_raises_for_small_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            load_data(n_samples=1)


if __name__ == "__main__":
    unittest.main()
