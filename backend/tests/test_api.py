"""
Focused test suite for AI-Powered Air Quality Prediction & Decision Support System API.

Tests cover:
1. GET /health returns successful status when model loads.
2. GET /model-info returns features, genuine feature importances, and training maxima.
3. POST /predict returns finite prediction, category, recommendations, and warnings.
4. Negative pollutant value is rejected with clear validation error (422).
5. Input exceeding training maximum returns prediction with out-of-range warning.
6. AQI category-boundary behavior, clamping for negative values, and above-500 handling.
"""

import math
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.decision_support import classify_aqi

client = TestClient(app)


def test_health_endpoint():
    """1. GET /health returns a successful response when the model loads."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info_endpoint():
    """2. GET /model-info returns features, genuine feature importances and training maxima."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()

    # Features check
    assert data["features"] == ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]

    # Feature importances check (genuine Gini scores from fitted Random Forest)
    importances = data["feature_importances"]
    assert len(importances) == 6
    assert [item["feature"] for item in importances] == ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
    total_importance = sum(item["importance"] for item in importances)
    assert 0.99 <= total_importance <= 1.01  # Sums to 1.0

    # Training maxima check
    maxima = data["training_maxima"]
    assert "PM2.5" in maxima and maxima["PM2.5"] > 0
    assert "PM10" in maxima and maxima["PM10"] > 0

    # Metrics and cautionary note check
    assert data["metrics"]["r2"] > 0.8
    assert "feature_importance_note" in data
    assert "does not establish that a pollutant causes" in data["feature_importance_note"]


def test_valid_predict_request():
    """3. Valid POST /predict request returns a finite prediction, category, recommendations, and warnings."""
    # Visakhapatnam test example
    payload = {
        "pm25": 33.17,
        "pm10": 108.22,
        "no2": 42.45,
        "co": 0.73,
        "so2": 13.65,
        "o3": 34.85
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert math.isfinite(data["predicted_aqi"])
    assert data["predicted_aqi"] > 0
    assert data["category"] == "Moderate"  # Visakhapatnam prediction ~112.63 falls in 101-200 Moderate band
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0
    assert isinstance(data["warnings"], list)
    assert len(data["warnings"]) == 0  # In-range input produces empty warnings list
    assert data["input_values"]["pm25"] == 33.17
    assert data["model_name"] == "RandomForestRegressor"


def test_negative_pollutant_rejected():
    """4. A negative pollutant value is rejected with a clear validation error."""
    payload = {
        "pm25": -15.5,
        "pm10": 108.22,
        "no2": 42.45,
        "co": 0.73,
        "so2": 13.65,
        "o3": 34.85
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    err_str = str(data)
    assert "pm25" in err_str
    assert "non-negative" in err_str or "greater than or equal to 0" in err_str


def test_out_of_range_input_warning():
    """5. Input exceeding its training maximum returns a prediction and a non-empty warning."""
    # Training maximum for PM2.5 is 949.99 ug/m3
    payload = {
        "pm25": 1250.0,
        "pm10": 108.22,
        "no2": 42.45,
        "co": 0.73,
        "so2": 13.65,
        "o3": 34.85
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Model still returns a finite prediction without rejecting the request
    assert math.isfinite(data["predicted_aqi"])
    assert len(data["warnings"]) >= 1
    assert any("PM2.5" in w and "exceeds the maximum observed in the training data" in w for w in data["warnings"])


def test_aqi_category_boundaries_and_clamping():
    """6. AQI category-boundary behaviour, clamping for negative values, and above-500 handling."""
    # A. Negative estimate handling (clamped to 0.0, Good, warning emitted)
    disp, cat, warns = classify_aqi(-12.4)
    assert disp == 0.0
    assert cat == "Good"
    assert len(warns) == 1
    assert "negative estimate" in warns[0]
    assert "clamped to 0.0" in warns[0]

    # B. Above-500 estimate handling (unclamped numeric value visible, Severe, warning emitted)
    disp, cat, warns = classify_aqi(542.8)
    assert disp == 542.8
    assert cat == "Severe"
    assert len(warns) == 1
    assert "exceeds the supported category range" in warns[0]

    # C. Exact boundary transitions:
    # 0 - 50: Good
    disp, cat, warns = classify_aqi(0.0)
    assert cat == "Good" and len(warns) == 0

    disp, cat, warns = classify_aqi(50.0)
    assert cat == "Good" and len(warns) == 0

    disp, cat, warns = classify_aqi(50.49)
    assert cat == "Good" and len(warns) == 0

    # 51 - 100: Satisfactory
    disp, cat, warns = classify_aqi(50.5)
    assert cat == "Satisfactory" and len(warns) == 0

    disp, cat, warns = classify_aqi(100.0)
    assert cat == "Satisfactory" and len(warns) == 0

    # 101 - 200: Moderate
    disp, cat, warns = classify_aqi(100.5)
    assert cat == "Moderate" and len(warns) == 0

    disp, cat, warns = classify_aqi(200.0)
    assert cat == "Moderate" and len(warns) == 0

    # 201 - 300: Poor
    disp, cat, warns = classify_aqi(200.5)
    assert cat == "Poor" and len(warns) == 0

    disp, cat, warns = classify_aqi(300.0)
    assert cat == "Poor" and len(warns) == 0

    # 301 - 400: Very Poor
    disp, cat, warns = classify_aqi(300.5)
    assert cat == "Very Poor" and len(warns) == 0

    disp, cat, warns = classify_aqi(400.0)
    assert cat == "Very Poor" and len(warns) == 0

    # 401 - 500: Severe
    disp, cat, warns = classify_aqi(400.5)
    assert cat == "Severe" and len(warns) == 0

    disp, cat, warns = classify_aqi(500.0)
    assert cat == "Severe" and len(warns) == 0
