
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.exists("student_result_model.pkl")
        )

    def test_model_can_be_loaded(self):
        model = joblib.load("student_result_model.pkl")
        self.assertIsNotNone(model)

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_meets_threshold(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(
            metrics["accuracy"], 0.85
        )

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_dataset_has_required_columns(self):
        data = pd.read_csv("student_results.csv")

        self.assertTrue(
            {"study_hours", "attendance", "result"}
            .issubset(data.columns)
        )


if __name__ == "__main__":
    unittest.main()
