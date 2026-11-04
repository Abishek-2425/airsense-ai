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
from datetime import datetime

# Ensure backend and preprocessor modules are resolvable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

# Register preprocessor in sys.modules so joblib resolves cleanly
try:
    from backend import preprocessor
    sys.modules['preprocessor'] = preprocessor
    from backend.preprocessor import (
        AirSenseCompositeModel, AirSensePreprocessor, FEATURE_ORDER,
        CO_ANOMALY_THRESHOLD, AQI_MAX_STATUTORY, _OMITTED, validate_month_integer
    )
except ImportError:
    import preprocessor
    sys.modules['preprocessor'] = preprocessor
    from preprocessor import (
        AirSenseCompositeModel, AirSensePreprocessor, FEATURE_ORDER,
        CO_ANOMALY_THRESHOLD, AQI_MAX_STATUTORY, _OMITTED, validate_month_integer
    )

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
    if not metrics_block:
        raise MetadataLoadError("Evaluation metrics block not found in model_metadata.json.")

    # Safe metric extraction without eager KeyError
    r2_val = None
    for k in ('all_cities_r2', 'r2'):
        if k in metrics_block:
            r2_val = metrics_block[k]
            break

    rmse_val = None
    for k in ('all_cities_rmse', 'rmse'):
        if k in metrics_block:
            rmse_val = metrics_block[k]
            break

    mae_val = None
    for k in ('all_cities_mae', 'mae'):
        if k in metrics_block:
            mae_val = metrics_block[k]
            break

    samples_val = None
    for k in ('sample_count', 'test_samples', 'n_samples'):
        if k in metrics_block:
            samples_val = metrics_block[k]
            break

    # Strict numeric validation
    for name, val in [("r2", r2_val), ("rmse", rmse_val), ("mae", mae_val)]:
        if val is None:
            raise MetadataLoadError(f"Missing required evaluation metric '{name}' in metadata.")
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise MetadataLoadError(f"Metric '{name}' must be numeric (received: {val}).")
        if np.isnan(val) or np.isinf(val):
            raise MetadataLoadError(f"Metric '{name}' cannot be NaN or infinite ({val}).")

    if r2_val > 1.0:
        raise MetadataLoadError(f"R² cannot exceed 1.0 (received: {r2_val}).")
    if rmse_val < 0.0:
        raise MetadataLoadError(f"RMSE cannot be negative (received: {rmse_val}).")
    if mae_val < 0.0:
        raise MetadataLoadError(f"MAE cannot be negative (received: {mae_val}).")

    if samples_val is None:
        raise MetadataLoadError("Missing 'sample_count' in metadata metrics block.")
    if isinstance(samples_val, bool) or not isinstance(samples_val, (int, np.integer)):
        raise MetadataLoadError(f"Sample count must be a genuine integer (received: {samples_val}).")
    if samples_val <= 0:
        raise MetadataLoadError(f"Sample count must be positive (received: {samples_val}).")

    return {
        "model_name": metadata.get('model_name', "AirSense AI Release Candidate v2.0"),
        "model_version": metadata.get('model_version', "2.0.0"),
        "target": "Next-calendar-day AQI",
        "features": metadata.get('feature_order', FEATURE_COLUMNS),
        "feature_importances": importances,
        "training_maxima": maxima,
        "metrics": {
            "r2": round(float(r2_val), 4),
            "rmse": round(float(rmse_val), 2),
            "mae": round(float(mae_val), 2),
            "test_samples": int(samples_val)
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
    month: Any = None
) -> Dict[str, Any]:
    """
    High-level operational prediction function.
    Implements monthly particulate ratio estimation, CO anomaly screening,
    raw prediction, and operational display clamping.
    """
    model = load_pipeline()

    if month is None or month is _OMITTED:
        validated_month = datetime.now().month
    else:
        validated_month = validate_month_integer(month, allow_none=False)

    if hasattr(model, 'predict_single'):
        return model.predict_single(
            pm25=pm25, pm10=pm10, no2=no2, co=co, so2=so2, o3=o3, month=validated_month
        )
    else:
        # Standard legacy scikit-learn pipeline fallback:
        # Pass original six pollutant inputs directly, including NaNs where missing.
        # Do not run AirSensePreprocessor; allow the pipeline's own fitted imputer to handle missing values.
        for name, val in [('NO2', no2), ('CO', co), ('SO2', so2), ('O3', o3)]:
            if val is None or not np.isfinite(val):
                raise ValueError(f"Pollutant '{name}' must be a finite number.")
            if val < 0.0:
                raise ValueError(f"Pollutant '{name}' cannot be negative ({val}).")

        for name, val in [('PM2.5', pm25), ('PM10', pm10)]:
            if val is not None:
                if not np.isfinite(val):
                    raise ValueError(f"Pollutant '{name}' must be a finite number.")
                if val < 0.0:
                    raise ValueError(f"Pollutant '{name}' cannot be negative ({val}).")

        warnings = []
        if co > CO_ANOMALY_THRESHOLD:
            warnings.append(
                f"Carbon Monoxide (CO={co:.2f} mg/m³) exceeds the project-defined CO screening threshold "
                f"({CO_ANOMALY_THRESHOLD:.1f} mg/m³). Model predictions may have increased uncertainty "
                f"above this level."
            )
        if pm10 is None:
            warnings.append("PM10 was not provided. Pipeline internal imputation applied.")
        if pm25 is None:
            warnings.append("PM2.5 was not provided. Pipeline internal imputation applied.")

        # Build 1x6 DataFrame with genuine NaNs for missing particulates
        df_input = pd.DataFrame(
            [[
                np.nan if pm25 is None else float(pm25),
                np.nan if pm10 is None else float(pm10),
                float(no2),
                float(co),
                float(so2),
                float(o3)
            ]],
            columns=FEATURE_COLUMNS
        )

        raw_pred = float(model.predict(df_input)[0])
        clamped_pred = float(np.clip(raw_pred, 0.0, 500.0))

        if clamped_pred <= 50: category = "Good"
        elif clamped_pred <= 100: category = "Satisfactory"
        elif clamped_pred <= 200: category = "Moderate"
        elif clamped_pred <= 300: category = "Poor"
        elif clamped_pred <= 400: category = "Very Poor"
        else: category = "Severe"

        if hasattr(model, 'named_steps') and 'imputer' in model.named_steps:
            transformed = model.named_steps['imputer'].transform(df_input)
            features_used = {name: round(float(transformed[0, i]), 2) for i, name in enumerate(FEATURE_COLUMNS)}
        else:
            features_used = {
                name: (round(float(df_input.iloc[0, i]), 2) if pd.notnull(df_input.iloc[0, i]) else None)
                for i, name in enumerate(FEATURE_COLUMNS)
            }

        return {
            'raw_predicted_aqi': round(raw_pred, 2),
            'predicted_aqi': round(clamped_pred, 2),
            'aqi_category': category,
            'warnings': warnings,
            'features_used': features_used,
            'model_version': getattr(model, 'version', "1.0.0")
        }


def predict_aqi_raw(ordered_features: Dict[str, float]) -> float:
    """
    Legacy prediction interface for backward compatibility.
    Branches cleanly between composite models (which require month handling)
    and standard scikit-learn pipelines (which expect strictly the 6 pollutant features).
    """
    model = load_pipeline()

    # Strict month validation if caller provides 'Month'
    month_val = ordered_features.get('Month', ordered_features.get('month'))
    if month_val is not None:
        month_int = validate_month_integer(month_val, allow_none=False)
    else:
        month_int = datetime.now().month

    if hasattr(model, 'predict_single'):
        pred_dict = model.predict_single(
            pm25=ordered_features.get('PM2.5'),
            pm10=ordered_features.get('PM10'),
            no2=ordered_features['NO2'],
            co=ordered_features['CO'],
            so2=ordered_features['SO2'],
            o3=ordered_features['O3'],
            month=month_int
        )
        return float(pred_dict['raw_predicted_aqi'])
    else:
        # Standard pipeline expects strictly the 6 pollutant features
        feature_vals = [[
            ordered_features.get(col, np.nan) for col in FEATURE_COLUMNS
        ]]
        df_input = pd.DataFrame(feature_vals, columns=FEATURE_COLUMNS)
        prediction = model.predict(df_input)
        return float(prediction[0])
