from __future__ import annotations

import unittest

import pandas as pd

from src.preprocess import split_and_scale


class PreprocessTests(unittest.TestCase):
    def test_split_and_scale_returns_expected_shapes(self) -> None:
        dataframe = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            }
        )

        x_train, x_test, y_train, y_test, scaler = split_and_scale(dataframe, test_size=0.4, random_state=42)

        self.assertEqual(x_train.shape, (3, 2))
        self.assertEqual(x_test.shape, (2, 2))
        self.assertEqual(len(y_train), 3)
        self.assertEqual(len(y_test), 2)
        self.assertIsNotNone(scaler)

    def test_split_and_scale_raises_for_missing_target(self) -> None:
        dataframe = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0]})

        with self.assertRaises(ValueError):
            split_and_scale(dataframe, target_column="target")

    def test_split_and_scale_raises_for_invalid_test_size(self) -> None:
        dataframe = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0],
                "target": [10.0, 11.0, 12.0],
            }
        )

        with self.assertRaises(ValueError):
            split_and_scale(dataframe, test_size=1.0)


if __name__ == "__main__":
    unittest.main()
