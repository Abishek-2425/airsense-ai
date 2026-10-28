"""
Automated Test Suite for AirSense AI Candidate Model & Validation Pipeline.

Tests verify:
1. Date separation, temporal embargo, and zero calendar date overlap.
2. Feature order integrity across dictionary and DataFrame inputs.
3. Target alignment (strictly next calendar day t+1).
4. Missing-value handling across all three hierarchical levels.
5. Reproducibility of candidate model predictions from disk.
6. Safety assertion that production aqi_model.pkl remains untouched.
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "archive" / "city_day.csv"
CURRENT_MODEL_PATH = PROJECT_ROOT / "aqi_model.pkl"
CANDIDATE_MODEL_PATH = PROJECT_ROOT / "model_validation" / "aqi_model_candidate.pkl"
METADATA_PATH = PROJECT_ROOT / "model_validation" / "candidate_metadata.json"

FEATURE_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
TARGET_COLUMN = 'Target_AQI'

from model_validation.train_candidate_model import (
    load_and_pair_data,
    split_data_with_embargo,
    TRAIN_MAX_DATE,
    EMBARGO_DATE,
    TEST_MIN_DATE
)
from model_validation.imputer import CityMonthAwareImputer


def test_date_separation_and_embargo():
    """1. Verify explicit date cutoffs, 1-day embargo, and zero date overlap."""
    paired_df = load_and_pair_data(str(CSV_PATH))
    train_df, embargo_df, test_df = split_data_with_embargo(paired_df)

    # Max train date <= 2019-12-04
    assert train_df['Date'].max() <= TRAIN_MAX_DATE, (
        f"Train max date {train_df['Date'].max()} exceeds {TRAIN_MAX_DATE}"
    )

    # Min test date >= 2019-12-06
    assert test_df['Date'].min() >= TEST_MIN_DATE, (
        f"Test min date {test_df['Date'].min()} is earlier than {TEST_MIN_DATE}"
    )

    # Embargo date is exactly 2019-12-05
    assert (embargo_df['Date'] == EMBARGO_DATE).all()

    # Zero date overlap between train and test
    train_dates = set(train_df['Date'])
    test_dates = set(test_df['Date'])
    assert len(train_dates.intersection(test_dates)) == 0, "Train and test sets share calendar dates!"

    # Embargo date does not exist in train or test
    assert EMBARGO_DATE not in train_dates
    assert EMBARGO_DATE not in test_dates

    # Train target dates do not cross into test observation dates
    assert train_df['Next_Date'].max() < test_df['Date'].min(), (
        f"Train target date {train_df['Next_Date'].max()} overlaps test date {test_df['Date'].min()}"
    )


def test_target_alignment():
    """2. Confirm that every target corresponds strictly to the next calendar day (t+1)."""
    paired_df = load_and_pair_data(str(CSV_PATH))
    train_df, _, test_df = split_data_with_embargo(paired_df)

    for name, df in [("Train", train_df), ("Test", test_df)]:
        diffs = df['Next_Date'] - df['Date']
        assert (diffs == pd.Timedelta(days=1)).all(), f"{name} set contains non-consecutive pairings!"
        assert df[TARGET_COLUMN].notnull().all(), f"{name} set contains null targets!"


def test_feature_order_integrity():
    """3. Confirm that the candidate pipeline enforces exact feature order."""
    assert CANDIDATE_MODEL_PATH.exists(), "Candidate model artifact not found!"
    candidate_pipe = joblib.load(CANDIDATE_MODEL_PATH)

    # Input dictionary in reversed order
    sample_reversed = {
        'O3': 34.85,
        'SO2': 13.65,
        'CO': 0.73,
        'NO2': 42.45,
        'PM10': 108.22,
        'PM2.5': 33.17,
    }
    df_rev = pd.DataFrame([sample_reversed])

    # Input dictionary in normal order
    sample_normal = {col: sample_reversed[col] for col in FEATURE_COLUMNS}
    df_norm = pd.DataFrame([sample_normal])

    pred_rev = candidate_pipe.predict(df_rev)
    pred_norm = candidate_pipe.predict(df_norm)

    # Predictions must be identical regardless of input column ordering
    np.testing.assert_allclose(pred_rev, pred_norm, rtol=1e-5)


def test_missing_value_handling_and_fallback():
    """4. Verify hierarchical missing value imputation across all 3 fallback levels."""
    # Build toy dataset to verify hierarchical fallback
    train_data = pd.DataFrame({
        'City': ['Delhi', 'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Bengaluru', 'Bengaluru', 'Bengaluru', 'Bengaluru', 'Bengaluru'],
        'Month': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'PM2.5': [100.0, 110.0, 90.0, 120.0, 80.0, 30.0, 35.0, 25.0, 40.0, 20.0],
        'PM10': [200.0, 210.0, 190.0, 220.0, 180.0, 60.0, 70.0, 50.0, 80.0, 40.0],
        'NO2': [40.0, 45.0, 35.0, 50.0, 30.0, 15.0, 18.0, 12.0, 20.0, 10.0],
        'CO': [2.0, 2.2, 1.8, 2.5, 1.5, 0.8, 0.9, 0.7, 1.0, 0.6],
        'SO2': [15.0, 16.0, 14.0, 17.0, 13.0, 8.0, 9.0, 7.0, 10.0, 6.0],
        'O3': [30.0, 32.0, 28.0, 35.0, 25.0, 20.0, 22.0, 18.0, 25.0, 15.0],
    })

    imputer = CityMonthAwareImputer(min_samples=5)
    imputer.fit(train_data)

    # Level 1: Known City + Month with missing PM10
    test_l1 = pd.DataFrame({
        'City': ['Delhi'],
        'Month': [1],
        'PM2.5': [100.0],
        'PM10': [np.nan],  # Missing
        'NO2': [40.0],
        'CO': [2.0],
        'SO2': [15.0],
        'O3': [30.0],
    })
    res_l1 = imputer.transform(test_l1)
    # Delhi Jan median PM10 is 200.0
    assert res_l1[0, 1] == 200.0

    # Level 2: Known City, unknown Month (e.g. Month 7)
    test_l2 = pd.DataFrame({
        'City': ['Delhi'],
        'Month': [7],
        'PM2.5': [100.0],
        'PM10': [np.nan],
        'NO2': [40.0],
        'CO': [2.0],
        'SO2': [15.0],
        'O3': [30.0],
    })
    res_l2 = imputer.transform(test_l2)
    # Falls back to Delhi city median (200.0)
    assert res_l2[0, 1] == 200.0

    # Level 3: Unknown / Zero-shot City (e.g. "Kochi")
    test_l3 = pd.DataFrame({
        'City': ['Kochi'],
        'Month': [1],
        'PM2.5': [np.nan],
        'PM10': [np.nan],
        'NO2': [np.nan],
        'CO': [np.nan],
        'SO2': [np.nan],
        'O3': [np.nan],
    })
    res_l3 = imputer.transform(test_l3)
    # Falls back to global training medians
    assert not np.isnan(res_l3).any()
    assert res_l3[0, 0] == float(train_data['PM2.5'].median())

    # Level 4: Standalone array / dict without City/Month metadata (Production API mode)
    test_standalone = pd.DataFrame([{
        'PM2.5': 50.0,
        'PM10': np.nan,  # Missing
        'NO2': 25.0,
        'CO': 1.0,
        'SO2': 10.0,
        'O3': 30.0
    }])
    res_standalone = imputer.transform(test_standalone)
    assert not np.isnan(res_standalone).any()
    assert res_standalone[0, 1] == float(train_data['PM10'].median())


def test_candidate_reproducibility():
    """5. Verify that the candidate model produces deterministic, reproducible predictions."""
    candidate_pipe1 = joblib.load(CANDIDATE_MODEL_PATH)
    candidate_pipe2 = joblib.load(CANDIDATE_MODEL_PATH)

    benchmark_input = pd.DataFrame([{
        'PM2.5': 33.17,
        'PM10': 108.22,
        'NO2': 42.45,
        'CO': 0.73,
        'SO2': 13.65,
        'O3': 34.85,
        'City': 'Visakhapatnam',
        'Date': '2020-06-22'
    }])

    pred1 = candidate_pipe1.predict(benchmark_input)
    pred2 = candidate_pipe2.predict(benchmark_input)

    assert pred1[0] == pred2[0]
    assert np.isfinite(pred1[0])


def test_production_model_safety():
    """6. STRICT SAFETY TEST: Ensure production aqi_model.pkl is intact and separate."""
    assert CURRENT_MODEL_PATH.exists(), "Production aqi_model.pkl is missing!"
    assert CANDIDATE_MODEL_PATH.exists(), "Candidate aqi_model_candidate.pkl is missing!"

    # Ensure candidate path is within model_validation/
    assert "model_validation" in str(CANDIDATE_MODEL_PATH)

    # Ensure paths are distinct
    assert CURRENT_MODEL_PATH != CANDIDATE_MODEL_PATH

    # Verify that production model can still be loaded and executed independently
    prod_model = joblib.load(CURRENT_MODEL_PATH)
    prod_pred = prod_model.predict(pd.DataFrame([[33.17, 108.22, 42.45, 0.73, 13.65, 34.85]], columns=FEATURE_COLUMNS))
    assert np.isfinite(prod_pred[0])
    # The production model benchmark prediction is ~112.63
    assert abs(prod_pred[0] - 112.63) < 0.1
