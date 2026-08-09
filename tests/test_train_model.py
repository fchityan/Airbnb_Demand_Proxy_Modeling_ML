from __future__ import annotations

import unittest

from src.data_loader import load_data
from src.preprocess import split_and_scale
from src.train_model import (
    build_feature_importance_for_models,
    predict_mean_baseline,
    train_models,
)


class TrainModelTests(unittest.TestCase):
    def setUp(self) -> None:
        dataframe = load_data(random_state=42, n_samples=200)
        self.x_train, self.x_test, self.y_train, _, _ = split_and_scale(dataframe)

    def test_train_models_returns_expected_keys(self) -> None:
        models = train_models(self.x_train, self.y_train, random_state=42)
        self.assertIn("linear_regression", models)
        self.assertIn("xgboost", models)

    def test_predict_mean_baseline_is_constant(self) -> None:
        prediction = predict_mean_baseline(self.y_train, sample_count=len(self.x_test))
        self.assertEqual(len(prediction), len(self.x_test))
        self.assertTrue((prediction == prediction[0]).all())

    def test_feature_importance_contains_expected_columns(self) -> None:
        models = train_models(self.x_train, self.y_train, random_state=42)
        importance = build_feature_importance_for_models(models, list(self.x_train.columns))

        self.assertIn("model", importance.columns)
        self.assertIn("feature", importance.columns)
        self.assertIn("importance", importance.columns)
        self.assertIn("coefficient", importance.columns)
        self.assertIn("linear_regression", set(importance["model"]))
        self.assertIn("xgboost", set(importance["model"]))


if __name__ == "__main__":
    unittest.main()