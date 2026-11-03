"""
Automated Backend API and ML Regression Test Suite for AirSense AI v2.0.
"""

import unittest
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.predictor import load_pipeline, load_metadata

client = TestClient(app)


class TestBackendAPI(unittest.TestCase):

    def test_01_health_endpoint(self):
        """Verify /health reports status 'ok' and model_loaded True."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_version"], "2.0.0")

    def test_02_model_info_endpoint(self):
        """Verify /model-info returns accurate architecture, metrics, and feature importances."""
        response = client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_version"], "2.0.0")
        self.assertEqual(data["features"], ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"])

        # Check feature importances
        importances = {item["feature"]: item["importance"] for item in data["feature_importances"]}
        self.assertIn("PM2.5", importances)
        self.assertIn("PM10", importances)
        self.assertIn("CO", importances)

        # Verify particulate dominance and CO reduction
        particulates = importances["PM2.5"] + importances["PM10"]
        self.assertGreater(particulates, 0.80, f"Expected particulates > 80%, got {particulates*100:.2f}%")
        self.assertLess(importances["CO"], 0.10, f"Expected CO < 10%, got {importances['CO']*100:.2f}%")

        # Verify metrics
        self.assertGreater(data["metrics"]["r2"], 0.85)
        self.assertLess(data["metrics"]["mae"], 25.0)

    def test_03_example_endpoint(self):
        """Verify /example returns historical reference sample."""
        response = client.get("/example")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["city"], "Visakhapatnam")
        self.assertIn("input_values", data)

    def test_04_predict_full_payload(self):
        """Verify /predict succeeds when all 6 pollutants are provided."""
        payload = {
            "pm25": 45.0,
            "pm10": 90.0,
            "no2": 25.0,
            "co": 1.1,
            "so2": 8.5,
            "o3": 30.0,
            "month": 5
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_aqi", data)
        self.assertIn("raw_predicted_aqi", data)
        self.assertIn("category", data)
        self.assertIn("recommendations", data)
        self.assertTrue(0.0 <= data["predicted_aqi"] <= 500.0)
        self.assertEqual(data["model_version"], "2.0.0")

    def test_05_predict_missing_pm10_ratio_imputation(self):
        """Verify /predict safely estimates missing PM10 using physical monthly ratio."""
        payload = {
            "pm25": 50.0,
            # pm10 omitted
            "no2": 25.0,
            "co": 1.1,
            "so2": 8.5,
            "o3": 30.0,
            "month": 1 # Winter ratio (~0.56)
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_aqi", data)
        # Verify warning emitted
        self.assertTrue(any("PM10 was not provided" in w for w in data["warnings"]))
        # Verify estimated PM10 is reasonable (50 / ~0.56 ≈ 89.16)
        features_used = data["features_used"]
        self.assertIsNotNone(features_used)
        self.assertAlmostEqual(features_used["PM10"], 89.16, delta=5.0)

    def test_06_predict_missing_pm25_ratio_imputation(self):
        """Verify /predict safely estimates missing PM2.5 when PM10 is provided."""
        payload = {
            # pm25 omitted
            "pm10": 100.0,
            "no2": 25.0,
            "co": 1.1,
            "so2": 8.5,
            "o3": 30.0,
            "month": 5 # Summer ratio (~0.40)
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(any("PM2.5 was not provided" in w for w in data["warnings"]))
        features_used = data["features_used"]
        self.assertAlmostEqual(features_used["PM2.5"], 39.88, delta=5.0)

    def test_07_predict_co_anomaly_advisory(self):
        """Verify /predict accepts CO > 17 mg/m3 but emits anomaly warning."""
        payload = {
            "pm25": 50.0,
            "pm10": 90.0,
            "no2": 25.0,
            "co": 28.5, # > 17 mg/m3
            "so2": 8.5,
            "o3": 30.0
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(any("exceeds statutory ambient threshold" in w for w in data["warnings"]))

    def test_08_predict_negative_pollutant_rejected(self):
        """Verify negative pollutant values are strictly rejected with HTTP 422."""
        payload = {
            "pm25": -5.0,
            "pm10": 90.0,
            "no2": 25.0,
            "co": 1.1,
            "so2": 8.5,
            "o3": 30.0
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_09_predict_missing_required_gaseous_pollutant(self):
        """Verify omission of gaseous pollutants (NO2, CO, SO2, O3) is rejected with HTTP 422."""
        payload = {
            "pm25": 50.0,
            "pm10": 90.0,
            # no2 missing
            "co": 1.1,
            "so2": 8.5,
            "o3": 30.0
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_10_display_clamping_above_500(self):
        """Verify operational display predicted_aqi is clamped to 500 when raw output exceeds 500."""
        # Provide extreme inputs to push regression tree to maximum
        payload = {
            "pm25": 900.0,
            "pm10": 900.0,
            "no2": 350.0,
            "co": 17.0,
            "so2": 180.0,
            "o3": 250.0
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(data["predicted_aqi"], 500.0)
        self.assertEqual(data["category"], "Severe")


if __name__ == '__main__':
    unittest.main()

