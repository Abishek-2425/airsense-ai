"""
Pydantic schemas for the Air Quality Prediction & Decision Support System API.
"""

import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PredictRequest(BaseModel):
    """
    Request payload containing the six pollutant readings for current day.
    """
    model_config = ConfigDict(extra='forbid')

    pm25: float = Field(..., description="PM2.5 concentration in ug/m3")
    pm10: float = Field(..., description="PM10 concentration in ug/m3")
    no2: float = Field(..., description="NO2 concentration in ug/m3")
    co: float = Field(..., description="CO concentration in mg/m3")
    so2: float = Field(..., description="SO2 concentration in ug/m3")
    o3: float = Field(..., description="O3 concentration in ug/m3")

    @field_validator('pm25', 'pm10', 'no2', 'co', 'so2', 'o3')
    @classmethod
    def validate_pollutant_value(cls, v: float, info) -> float:
        if not math.isfinite(v):
            raise ValueError(f"Value for '{info.field_name}' must be a finite number (NaN and Infinity are not permitted).")
        if v < 0.0:
            raise ValueError(f"Value for '{info.field_name}' must be non-negative (received {v}).")
        return float(v)

    def to_feature_dict(self) -> Dict[str, float]:
        """
        Maps JSON request fields to model feature names in exact training order:
        ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3'].
        """
        return {
            'PM2.5': self.pm25,
            'PM10': self.pm10,
            'NO2': self.no2,
            'CO': self.co,
            'SO2': self.so2,
            'O3': self.o3
        }


class PredictResponse(BaseModel):
    """
    Response schema for next-calendar-day AQI prediction and decision support.
    """
    predicted_aqi: float = Field(..., description="Displayed predicted next-calendar-day AQI (clamped to 0 if negative)")
    raw_predicted_aqi: float = Field(..., description="Raw model regression output before display clamping")
    category: str = Field(..., description="AQI category assigned according to prototype bands")
    recommendations: List[str] = Field(..., description="Concise rule-based advisory recommendations")
    warnings: List[str] = Field(..., description="List of warning messages for out-of-range inputs or clamping")
    input_values: Dict[str, float] = Field(..., description="Submitted pollutant input readings")
    model_name: str = Field(..., description="Identifier of the model used for prediction")


class FeatureImportanceItem(BaseModel):
    """Pairing of feature name with its Gini/impurity feature importance score."""
    feature: str
    importance: float


class EvaluationMetrics(BaseModel):
    """Reported evaluation metrics on the matched chronological test set."""
    r2: float
    rmse: float
    mae: float
    test_samples: int


class ModelInfoResponse(BaseModel):
    """Detailed model architecture, metadata, feature importance, and performance metrics."""
    model_name: str
    target: str
    features: List[str]
    feature_importances: List[FeatureImportanceItem]
    training_maxima: Dict[str, float]
    metrics: EvaluationMetrics
    feature_importance_note: str
    dataset_summary: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Service health and model load readiness status."""
    status: str
    model_loaded: bool


class ExampleResponse(BaseModel):
    """Historical test example from training report (Visakhapatnam 2020-06-22)."""
    city: str
    date: str
    input_values: Dict[str, float]
    today_aqi: float
    actual_next_day_aqi: float
    reported_model_prediction: float
    reported_absolute_error: float
    note: Optional[str] = None
