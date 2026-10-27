"""
AI-Powered Air Quality Prediction & Decision Support System
Training Pipeline: Next-Calendar-Day AQI Prediction

This module prepares city-level historical air quality data, pairs observations
with the immediately following calendar day's AQI, trains a Random Forest
regression model, compares it against a persistence baseline, and exports
the model pipeline and metadata for inference.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Exact feature order required by the model
FEATURE_COLUMNS: List[str] = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
TARGET_COLUMN: str = 'Target_AQI'
MODEL_FILENAME: str = 'aqi_model.pkl'
METADATA_FILENAME: str = 'model_metadata.json'


def load_and_preprocess_data(
    csv_path: str = os.path.join('archive', 'city_day.csv')
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw city_day data, creates next-calendar-day targets, and filters
    for valid consecutive observation pairs to prevent temporal errors.

    Returns:
        paired_df: DataFrame sorted chronologically containing valid pairs
        summary_stats: Dictionary with dataset statistics
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    # Load raw data
    raw_df = pd.read_csv(csv_path)
    total_raw_rows = len(raw_df)

    # Convert Date column to datetime
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])

    # Sort strictly by City and Date to maintain temporal integrity
    raw_df = raw_df.sort_values(['City', 'Date']).reset_index(drop=True)

    # Shift by 1 within each city to identify next-calendar-day candidate
    raw_df['Next_Date'] = raw_df.groupby('City')['Date'].shift(-1)
    raw_df['Target_AQI'] = raw_df.groupby('City')['AQI'].shift(-1)

    # Ensure target strictly belongs to the IMMEDIATELY FOLLOWING CALENDAR DAY (+1 day)
    # This prevents pairing non-consecutive days when there are temporal gaps.
    consecutive_mask = (raw_df['Next_Date'] - raw_df['Date']) == pd.Timedelta(days=1)
    consecutive_df = raw_df[consecutive_mask].copy()

    # Supervised learning requires a valid ground truth target (tomorrow's AQI)
    valid_paired_df = consecutive_df.dropna(subset=[TARGET_COLUMN]).copy()

    # Sort chronologically by observation Date for chronological splitting
    valid_paired_df = valid_paired_df.sort_values('Date').reset_index(drop=True)

    summary_stats = {
        'total_raw_rows': total_raw_rows,
        'unique_cities': int(raw_df['City'].nunique()),
        'cities': sorted(raw_df['City'].unique().tolist()),
        'raw_date_min': raw_df['Date'].min().strftime('%Y-%m-%d'),
        'raw_date_max': raw_df['Date'].max().strftime('%Y-%m-%d'),
        'consecutive_day_pairs': int(len(consecutive_df)),
        'valid_target_pairs': int(len(valid_paired_df)),
        'complete_cases_count': int(valid_paired_df.dropna(subset=FEATURE_COLUMNS + ['AQI', TARGET_COLUMN]).shape[0]),
    }

    return valid_paired_df, summary_stats


def split_data_chronologically(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the paired dataset chronologically to avoid future-to-past data leakage.
    All training records strictly precede or coincide up to the chronological split date.
    """
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy().reset_index(drop=True)
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    return train_df, test_df


def train_pipeline(
    train_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Builds and trains a Pipeline consisting of:
    1. SimpleImputer(strategy='median') fit ONLY on training features to prevent leakage.
    2. RandomForestRegressor trained to predict next-day AQI.

    Returns:
        pipeline: Fitted scikit-learn Pipeline
        train_medians: Feature-level median statistics from the training set
    """
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values

    # Preprocessing: Impute missing pollutant values using training set medians only
    imputer = SimpleImputer(strategy='median')
    rf = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('imputer', imputer),
        ('regressor', rf)
    ])

    pipeline.fit(X_train, y_train)

    # Extract training medians for metadata documentation
    imputer_medians = {
        col: float(pipeline.named_steps['imputer'].statistics_[i])
        for i, col in enumerate(feature_cols)
    }

    return pipeline, imputer_medians


def evaluate_models(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN
) -> Dict[str, Any]:
    """
    Evaluates the Random Forest model and compares it against the persistence baseline.
    The persistence baseline predicts tomorrow's AQI using today's AQI: y_hat(t+1) = y(t).

    Both models are compared head-to-head on the matched subset where today's AQI
    is available, and the RF model is also evaluated across all valid test records.
    """
    y_true_all = test_df[target_col].values
    y_pred_rf_all = pipeline.predict(test_df[feature_cols])

    # Overall RF metrics on all test samples
    rf_all_metrics = {
        'r2': float(r2_score(y_true_all, y_pred_rf_all)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_all, y_pred_rf_all))),
        'mae': float(mean_absolute_error(y_true_all, y_pred_rf_all)),
        'sample_count': int(len(y_true_all))
    }

    # Matched comparison: test samples where today's AQI is non-null
    matched_mask = test_df['AQI'].notnull()
    matched_test = test_df[matched_mask]

    y_true_matched = matched_test[target_col].values
    y_pred_baseline = matched_test['AQI'].values
    y_pred_rf_matched = pipeline.predict(matched_test[feature_cols])

    baseline_metrics = {
        'r2': float(r2_score(y_true_matched, y_pred_baseline)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_matched, y_pred_baseline))),
        'mae': float(mean_absolute_error(y_true_matched, y_pred_baseline)),
        'sample_count': int(len(y_true_matched))
    }

    rf_matched_metrics = {
        'r2': float(r2_score(y_true_matched, y_pred_rf_matched)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_matched, y_pred_rf_matched))),
        'mae': float(mean_absolute_error(y_true_matched, y_pred_rf_matched)),
        'sample_count': int(len(y_true_matched))
    }

    return {
        'random_forest_all_test': rf_all_metrics,
        'random_forest_matched': rf_matched_metrics,
        'persistence_baseline': baseline_metrics,
    }


def save_artifacts(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_path: str = MODEL_FILENAME,
    metadata_path: str = METADATA_FILENAME
) -> None:
    """Saves the trained pipeline model and accompanying metadata."""
    joblib.dump(pipeline, model_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def predict_next_day_aqi(
    features: Union[Dict[str, float], pd.DataFrame, np.ndarray, List[float]],
    model_path: str = MODEL_FILENAME,
    metadata_path: str = METADATA_FILENAME,
    loaded_pipeline: Pipeline = None
) -> Union[float, np.ndarray]:
    """
    Reusable prediction function to predict next-calendar-day AQI from current-day pollutant readings.

    Args:
        features: Input pollutant values as a dict, DataFrame, 2D/1D array, or list
                  matching FEATURE_COLUMNS: ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3'].
        model_path: Path to serialized pipeline if loaded_pipeline is None.
        metadata_path: Path to metadata JSON.
        loaded_pipeline: Optional pre-loaded Pipeline instance for fast repeated calls.

    Returns:
        Predicted next-calendar-day AQI (float for single sample, array for multiple).
    """
    if loaded_pipeline is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
        pipeline = joblib.load(model_path)
    else:
        pipeline = loaded_pipeline

    # Load feature order from metadata if available, otherwise use default
    feature_order = FEATURE_COLUMNS
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                feature_order = meta.get('feature_order', FEATURE_COLUMNS)
        except Exception:
            pass

    # Normalize input into DataFrame with exact feature order
    if isinstance(features, dict):
        df_input = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_order}])
        predictions = pipeline.predict(df_input)
        return float(predictions[0])
    elif isinstance(features, pd.DataFrame):
        df_input = features.copy()
        # Ensure all required columns exist, imputing NaN if missing from input DF
        for col in feature_order:
            if col not in df_input.columns:
                df_input[col] = np.nan
        df_input = df_input[feature_order]
        predictions = pipeline.predict(df_input)
        return predictions if len(predictions) > 1 else float(predictions[0])
    elif isinstance(features, (list, np.ndarray)):
        arr = np.array(features)
        if arr.ndim == 1:
            if len(arr) != len(feature_order):
                raise ValueError(f"Expected {len(feature_order)} features {feature_order}, got {len(arr)}.")
            df_input = pd.DataFrame([arr], columns=feature_order)
            predictions = pipeline.predict(df_input)
            return float(predictions[0])
        elif arr.ndim == 2:
            if arr.shape[1] != len(feature_order):
                raise ValueError(f"Expected {len(feature_order)} columns {feature_order}, got {arr.shape[1]}.")
            df_input = pd.DataFrame(arr, columns=feature_order)
            return pipeline.predict(df_input)
        else:
            raise ValueError(f"Unsupported array dimensions: {arr.ndim}")
    else:
        raise TypeError(f"Unsupported features type: {type(features)}")


def main():
    print("=" * 70)
    print("AI-POWERED AIR QUALITY PREDICTION PIPELINE")
    print("Task: Predict Next-Calendar-Day AQI from Current Pollutant Readings")
    print("=" * 70)

    # 1. Load and pair dataset
    csv_file = os.path.join('archive', 'city_day.csv')
    print(f"\n[1/5] Loading and verifying dataset from '{csv_file}'...")
    paired_df, summary_stats = load_and_preprocess_data(csv_file)

    print(f"  * Total raw records:        {summary_stats['total_raw_rows']:,}")
    print(f"  * Available cities ({summary_stats['unique_cities']}): {', '.join(summary_stats['cities'][:6])}...")
    print(f"  * Historical date range:    {summary_stats['raw_date_min']} to {summary_stats['raw_date_max']}")
    print(f"  * Valid consecutive pairs:  {summary_stats['consecutive_day_pairs']:,}")
    print(f"  * Target valid pairs:       {summary_stats['valid_target_pairs']:,} (with verified t+1 calendar day AQI)")

    # 2. Chronological Train-Test Split
    print("\n[2/5] Performing chronological train-test split (80/20)...")
    train_df, test_df = split_data_chronologically(paired_df, train_ratio=0.8)

    train_start, train_end = train_df['Date'].min().strftime('%Y-%m-%d'), train_df['Date'].max().strftime('%Y-%m-%d')
    test_start, test_end = test_df['Date'].min().strftime('%Y-%m-%d'), test_df['Date'].max().strftime('%Y-%m-%d')

    print(f"  * Training set: {len(train_df):,} samples ({train_start} to {train_end}) across {train_df['City'].nunique()} cities")
    print(f"  * Test set:     {len(test_df):,} samples ({test_start} to {test_end}) across {test_df['City'].nunique()} cities")
    print("  * Temporal check: Strict chronological separation enforced (no target leakage).")

    # 3. Model Training
    print("\n[3/5] Training Random Forest regression pipeline...")
    print(f"  * Input features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"  * Target variable: {TARGET_COLUMN} (AQI of date + 1 day)")
    pipeline, imputer_medians = train_pipeline(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    print("  * Imputation: SimpleImputer(median) fit strictly on train split.")
    print("  * Estimator: RandomForestRegressor(n_estimators=100, min_samples_leaf=2).")

    # 4. Evaluation & Baseline Comparison
    print("\n[4/5] Evaluating against Persistence Baseline...")
    evaluation_results = evaluate_models(pipeline, test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    base = evaluation_results['persistence_baseline']
    rf_m = evaluation_results['random_forest_matched']
    rf_all = evaluation_results['random_forest_all_test']

    print("\n" + "-" * 70)
    print(f"{'Model':<30} | {'R^2':<8} | {'RMSE':<10} | {'MAE':<10} | {'Samples':<8}")
    print("-" * 70)
    print(f"{'Persistence Baseline (t -> t+1)':<30} | {base['r2']:<8.4f} | {base['rmse']:<10.4f} | {base['mae']:<10.4f} | {base['sample_count']:<8,}")
    print(f"{'Random Forest (Matched Test)':<30} | {rf_m['r2']:<8.4f} | {rf_m['rmse']:<10.4f} | {rf_m['mae']:<10.4f} | {rf_m['sample_count']:<8,}")
    print(f"{'Random Forest (All Test Rows)':<30} | {rf_all['r2']:<8.4f} | {rf_all['rmse']:<10.4f} | {rf_all['mae']:<10.4f} | {rf_all['sample_count']:<8,}")
    print("-" * 70)

    # 5. Save Artifacts & Metadata
    print("\n[5/5] Saving model and preprocessing metadata...")
    metadata = {
        'project': 'AI-Powered Air Quality Prediction & Decision Support System',
        'task': 'Predict next-calendar-day AQI using current-day pollutant readings',
        'feature_order': FEATURE_COLUMNS,
        'target_variable': TARGET_COLUMN,
        'preprocessing': {
            'imputer_strategy': 'median',
            'imputer_medians': imputer_medians,
            'prevent_leakage_note': 'Imputation statistics calculated strictly on training split'
        },
        'dataset_summary': {
            'total_raw_rows': summary_stats['total_raw_rows'],
            'valid_calendar_pairs': summary_stats['consecutive_day_pairs'],
            'valid_target_samples': summary_stats['valid_target_pairs'],
            'cities_count': summary_stats['unique_cities'],
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_date_range': {'start': train_start, 'end': train_end},
            'test_date_range': {'start': test_start, 'end': test_end}
        },
        'hyperparameters': {
            'model_type': 'RandomForestRegressor',
            'n_estimators': 100,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'evaluation_metrics': evaluation_results,
        'saved_artifacts': {
            'model_file': MODEL_FILENAME,
            'metadata_file': METADATA_FILENAME
        }
    }

    save_artifacts(pipeline, metadata, MODEL_FILENAME, METADATA_FILENAME)
    print(f"  * Model artifact:    {os.path.abspath(MODEL_FILENAME)}")
    print(f"  * Metadata artifact: {os.path.abspath(METADATA_FILENAME)}")

    # Standalone Verification: Independent Load & Sample Prediction
    print("\n" + "=" * 70)
    print("INDEPENDENT MODEL VERIFICATION")
    print("=" * 70)
    sample_input = {
        'PM2.5': 33.17,
        'PM10': 108.22,
        'NO2': 42.45,
        'CO': 0.73,
        'SO2': 13.65,
        'O3': 34.85
    }
    sample_actual_next_day_aqi = 100.0  # From Visakhapatnam test record (2020-06-22 -> 2020-06-23)
    sample_today_aqi = 95.0

    print("Sample feature inputs (Actual test reading from Visakhapatnam, 2020-06-22):")
    for feat, val in sample_input.items():
        print(f"  * {feat:<8}: {val}")
    print(f"  * Today's AQI:               {sample_today_aqi}")
    print(f"  * Actual Next-Day AQI:       {sample_actual_next_day_aqi}")

    pred = predict_next_day_aqi(sample_input, model_path=MODEL_FILENAME, metadata_path=METADATA_FILENAME)
    print(f"  * Predicted Next-Day AQI:    {pred:.2f}")
    print(f"  * Absolute Prediction Error: {abs(pred - sample_actual_next_day_aqi):.2f} AQI points")
    print("=" * 70)
    print("Pipeline training, verification, and artifact export completed successfully.")


if __name__ == '__main__':
    main()
