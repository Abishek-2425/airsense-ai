"""
Model Loader and Inference Predictor.

Loads the saved scikit-learn pipeline and metadata once and provides
thread-safe, deterministic prediction for the 6 target air pollutants.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Absolute paths resolved relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "aqi_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

# Strict feature order required by model
FEATURE_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']


class ModelLoadError(RuntimeError):
    """Raised when the serialized model artifact cannot be loaded or is invalid."""
    pass


class MetadataLoadError(RuntimeError):
    """Raised when the model metadata cannot be loaded or parsed."""
    pass


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    """
    Loads and caches the trained scikit-learn pipeline from aqi_model.pkl.
    Resolves the file path relative to the project root.
    """
    if not MODEL_PATH.exists():
        raise ModelLoadError(
            f"Model artifact not found at '{MODEL_PATH}'. "
            "Ensure aqi_model.pkl exists in the project root."
        )

    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        raise ModelLoadError(f"Failed to deserialize model artifact from '{MODEL_PATH}': {e}") from e

    # Verify pipeline structure
    if not isinstance(pipeline, Pipeline):
        raise ModelLoadError(f"Expected a scikit-learn Pipeline, got {type(pipeline).__name__}.")

    # Verify that the regressor step is a RandomForestRegressor
    if 'regressor' not in pipeline.named_steps:
        raise ModelLoadError("Pipeline is missing the expected 'regressor' step.")

    regressor = pipeline.named_steps['regressor']
    if not isinstance(regressor, RandomForestRegressor):
        raise ModelLoadError(f"Expected regressor to be RandomForestRegressor, got {type(regressor).__name__}.")

    # Verify feature_importances_ exists and is non-empty
    if not hasattr(regressor, 'feature_importances_'):
        raise ModelLoadError("Fitted regressor does not expose 'feature_importances_'.")

    # Ensure forward compatibility across scikit-learn versions for SimpleImputer
    imputer = pipeline.named_steps.get('imputer')
    if imputer is not None and not hasattr(imputer, '_fill_dtype') and hasattr(imputer, '_fit_dtype'):
        imputer._fill_dtype = imputer._fit_dtype

    return pipeline


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
    Extracts genuine Gini/impurity feature importance values from the
    trained RandomForestRegressor in the pipeline.
    """
    pipeline = load_pipeline()
    rf: RandomForestRegressor = pipeline.named_steps['regressor']
    raw_importances = rf.feature_importances_

    if len(raw_importances) != len(FEATURE_COLUMNS):
        raise ModelLoadError(
            f"Mismatch between number of features ({len(FEATURE_COLUMNS)}) "
            f"and feature importances ({len(raw_importances)})."
        )

    return [
        {"feature": feat, "importance": round(float(imp), 6)}
        for feat, imp in zip(FEATURE_COLUMNS, raw_importances)
    ]


def get_training_maxima() -> Dict[str, float]:
    """
    Retrieves the maximum non-null values observed during training
    for the six features from model_metadata.json.
    """
    metadata = load_metadata()
    maxima = metadata.get('training_maxima')
    if maxima is None:
        raise MetadataLoadError("'training_maxima' key not found in model_metadata.json.")
    return maxima


def get_model_info_dict() -> Dict[str, Any]:
    """
    Assembles comprehensive model metadata including architecture, metrics,
    genuine feature importances, and training bounds.
    """
    metadata = load_metadata()
    importances = get_feature_importances()
    maxima = get_training_maxima()

    # Extract metrics from matched evaluation if present
    metrics_block = metadata.get('evaluation_metrics', {}).get('random_forest_matched', {})
    if not metrics_block:
        metrics_block = metadata.get('evaluation_metrics', {}).get('random_forest_all_test', {})

    return {
        "model_name": metadata.get('hyperparameters', {}).get('model_type', 'RandomForestRegressor'),
        "target": "Next-calendar-day AQI",
        "features": metadata.get('feature_order', FEATURE_COLUMNS),
        "feature_importances": importances,
        "training_maxima": maxima,
        "metrics": {
            "r2": round(float(metrics_block.get('r2', 0.8367)), 4),
            "rmse": round(float(metrics_block.get('rmse', 36.46)), 2),
            "mae": round(float(metrics_block.get('mae', 20.10)), 2),
            "test_samples": int(metrics_block.get('sample_count', 4902))
        },
        "feature_importance_note": (
            "Feature importance shows how the trained model uses its inputs; "
            "it does not establish that a pollutant causes an AQI outcome."
        ),
        "dataset_summary": metadata.get('dataset_summary')
    }


def predict_aqi_raw(ordered_features: Dict[str, float]) -> float:
    """
    Executes raw prediction through the loaded scikit-learn pipeline.
    
    Args:
        ordered_features: Dictionary containing exactly the 6 required features:
                          ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3'].
    Returns:
        Raw floating-point regression prediction for next-day AQI.
    """
    pipeline = load_pipeline()

    # Build single-row DataFrame respecting exact column order
    df_input = pd.DataFrame([[ordered_features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    prediction = pipeline.predict(df_input)
    raw_val = float(prediction[0])

    if not np.isfinite(raw_val):
        raise RuntimeError("Model produced a non-finite prediction value.")

    return raw_val
