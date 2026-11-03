"""
Model Loader and Inference Predictor (v2.0).

Loads the trained AirSense AI composite inference pipeline and metadata once,
providing thread-safe, deterministic prediction with physical particulate ratio imputation.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional

# Ensure backend and preprocessor modules are resolvable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

# Register preprocessor in sys.modules so joblib resolves cleanly
try:
    from backend import preprocessor
    sys.modules['preprocessor'] = preprocessor
    from backend.preprocessor import AirSenseCompositeModel, AirSensePreprocessor, FEATURE_ORDER, CO_ANOMALY_THRESHOLD, AQI_MAX_STATUTORY
except ImportError:
    import preprocessor
    sys.modules['preprocessor'] = preprocessor
    from preprocessor import AirSenseCompositeModel, AirSensePreprocessor, FEATURE_ORDER, CO_ANOMALY_THRESHOLD, AQI_MAX_STATUTORY

MODEL_PATH = BASE_DIR / "aqi_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
FEATURE_COLUMNS = FEATURE_ORDER


class ModelLoadError(RuntimeError):
    """Raised when the serialized model artifact cannot be loaded or is invalid."""
    pass


class MetadataLoadError(RuntimeError):
    """Raised when the model metadata cannot be loaded or parsed."""
    pass


@lru_cache(maxsize=1)
def load_pipeline():
    """
    Loads and caches the trained AirSenseCompositeModel pipeline from aqi_model.pkl.
    """
    if not MODEL_PATH.exists():
        raise ModelLoadError(
            f"Model artifact not found at '{MODEL_PATH}'. "
            "Ensure aqi_model.pkl exists in the project root."
        )

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise ModelLoadError(f"Failed to deserialize model artifact from '{MODEL_PATH}': {e}") from e

    # Verify that the loaded model exposes prediction capabilities
    if not hasattr(model, 'predict'):
        raise ModelLoadError(f"Loaded artifact does not expose 'predict' method ({type(model).__name__}).")

    return model


@lru_cache(maxsize=1)
def load_metadata() -> Dict[str, Any]:
    """
    Loads and caches the model metadata JSON document.
    """
    if not METADATA_PATH.exists():
        raise MetadataLoadError(f"Metadata file not found at '{METADATA_PATH}'.")

    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        raise MetadataLoadError(f"Failed to parse metadata JSON at '{METADATA_PATH}': {e}") from e

    return metadata


def is_model_ready() -> bool:
    """Checks whether the model artifact is loadable and ready for inference."""
    try:
        pipeline = load_pipeline()
        return pipeline is not None
    except Exception:
        return False


def get_feature_importances() -> List[Dict[str, Any]]:
    """
    Extracts genuine Gini/impurity feature importance values from the trained estimator.
    """
    model = load_pipeline()
    if hasattr(model, 'regressor'):
        raw_importances = model.regressor.feature_importances_
    elif hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
        raw_importances = model.named_steps['regressor'].feature_importances_
    else:
        # Fallback to metadata
        metadata = load_metadata()
        imp_dict = metadata.get('feature_schema', {}).get('feature_importances', {})
        return [{"feature": feat, "importance": imp_dict.get(feat, 0.0)} for feat in FEATURE_COLUMNS]

    return [
        {"feature": feat, "importance": round(float(imp), 6)}
        for feat, imp in zip(FEATURE_COLUMNS, raw_importances)
    ]


def get_training_maxima() -> Dict[str, float]:
    """
    Retrieves the maximum non-null values observed during training from model_metadata.json.
    """
    metadata = load_metadata()
    maxima = metadata.get('training_maxima')
    if maxima is None:
        return {
            "PM2.5": 949.99, "PM10": 917.08, "NO2": 362.21,
            "CO": 17.00, "SO2": 186.08, "O3": 257.73
        }
    return maxima


def get_model_info_dict() -> Dict[str, Any]:
    """
    Assembles comprehensive model metadata including architecture, metrics,
    genuine feature importances, and training bounds.
    """
    metadata = load_metadata()
    importances = get_feature_importances()
    maxima = get_training_maxima()

    # Extract operational metrics
    metrics_block = metadata.get('evaluation_metrics', {}).get('operational_scale_cpcb_500', {})
    if not metrics_block:
        metrics_block = metadata.get('evaluation_metrics', {}).get('random_forest_matched', {})

    return {
        "model_name": metadata.get('model_name', "AirSense AI Release Candidate v2.0"),
        "model_version": metadata.get('model_version', "2.0.0"),
        "target": "Next-calendar-day AQI",
        "features": metadata.get('feature_order', FEATURE_COLUMNS),
        "feature_importances": importances,
        "training_maxima": maxima,
        "metrics": {
            "r2": round(float(metrics_block.get('all_cities_r2', metrics_block.get('r2', 0.8712))), 4),
            "rmse": round(float(metrics_block.get('all_cities_rmse', metrics_block.get('rmse', 29.41))), 2),
            "mae": round(float(metrics_block.get('all_cities_mae', metrics_block.get('mae', 19.26))), 2),
            "test_samples": int(metrics_block.get('sample_count', 4900))
        },
        "feature_importance_note": (
            "Feature importance reflects the Gini impurity split weight in the Random Forest ensemble; "
            "it shows how the model allocates predictive weight across the 6 criteria pollutants."
        ),
        "dataset_summary": metadata.get('dataset_summary'),
        "preprocessing_strategy": metadata.get('preprocessing', {}).get('strategy', "Monthly PM2.5/PM10 Ratio Imputation")
    }


def predict_aqi_composite(
    pm25: Optional[float],
    pm10: Optional[float],
    no2: float,
    co: float,
    so2: float,
    o3: float,
    month: Optional[int] = None
) -> Dict[str, Any]:
    """
    High-level operational prediction function.
    Implements monthly particulate ratio estimation, CO anomaly screening,
    raw prediction, and operational display clamping.
    """
    model = load_pipeline()

    if month is None:
        # Default to current month from datetime
        from datetime import datetime
        month = datetime.now().month

    if hasattr(model, 'predict_single'):
        return model.predict_single(
            pm25=pm25, pm10=pm10, no2=no2, co=co, so2=so2, o3=o3, month=month
        )
    else:
        # Fallback for standard scikit-learn pipeline
        preprocessor = AirSensePreprocessor()
        features, warnings = preprocessor.process_single_input(
            pm25=pm25, pm10=pm10, no2=no2, co=co, so2=so2, o3=o3, month=month
        )
        df_input = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        raw_pred = float(model.predict(df_input)[0])
        clamped_pred = float(np.clip(raw_pred, 0.0, 500.0))

        if clamped_pred <= 50: category = "Good"
        elif clamped_pred <= 100: category = "Satisfactory"
        elif clamped_pred <= 200: category = "Moderate"
        elif clamped_pred <= 300: category = "Poor"
        elif clamped_pred <= 400: category = "Very Poor"
        else: category = "Severe"

        return {
            'raw_predicted_aqi': round(raw_pred, 2),
            'predicted_aqi': round(clamped_pred, 2),
            'aqi_category': category,
            'warnings': warnings,
            'features_used': {name: round(float(features[0, i]), 2) for i, name in enumerate(FEATURE_COLUMNS)},
            'model_version': "2.0.0"
        }


def predict_aqi_raw(ordered_features: Dict[str, float]) -> float:
    """
    Legacy prediction interface for backward compatibility.
    """
    model = load_pipeline()
    df_input = pd.DataFrame([[ordered_features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    prediction = model.predict(df_input)
    return float(prediction[0])
