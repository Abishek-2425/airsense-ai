"""
Supporting script to compute non-null observed training maxima for the 6 pollutant features
from archive/city_day.csv without retraining or modifying aqi_model.pkl.
Updates model_metadata.json with the training_maxima field while preserving all existing keys.
"""

import os
import json
import pandas as pd
from train_aqi_model import (
    load_and_preprocess_data,
    split_data_chronologically,
    FEATURE_COLUMNS,
    METADATA_FILENAME
)


def compute_and_save_training_maxima(
    csv_path: str = os.path.join('archive', 'city_day.csv'),
    metadata_path: str = METADATA_FILENAME
):
    print("Loading data to compute training partition maxima...")
    paired_df, _ = load_and_preprocess_data(csv_path)
    train_df, _ = split_data_chronologically(paired_df, train_ratio=0.8)

    training_maxima = {}
    for col in FEATURE_COLUMNS:
        # Exclude missing (NaN) values and compute the maximum observed value
        max_val = float(train_df[col].dropna().max())
        training_maxima[col] = round(max_val, 2)

    print("Computed Training Maxima:")
    for col, val in training_maxima.items():
        print(f"  {col}: {val}")

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata['training_maxima'] = training_maxima
    metadata['training_maxima_note'] = (
        "Calculated as the maximum non-null observed value for each feature in the chronological training partition (2015-01-01 to 2019-12-05)."
    )

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated {metadata_path} successfully without modifying existing keys or retraining the model.")


if __name__ == '__main__':
    compute_and_save_training_maxima()
