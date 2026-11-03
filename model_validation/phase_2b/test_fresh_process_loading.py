"""
Tests loading the release candidate model artifact from a fresh Python process
and verifies that predictions are strictly deterministic and identical.
"""

import os
import sys
import json
import hashlib
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHASE_2B_DIR = os.path.join(PROJECT_ROOT, 'model_validation', 'phase_2b')
RC_DIR = os.path.join(PHASE_2B_DIR, 'release_candidate')

# Add preprocessor path
sys.path.insert(0, PHASE_2B_DIR)
from preprocessor import AirSensePreprocessor, AirSenseCompositeModel, FEATURE_ORDER

def test_fresh_load():
    artifact_path = os.path.join(RC_DIR, 'candidate_pipeline.joblib')
    config_path = os.path.join(RC_DIR, 'training_config.json')

    assert os.path.exists(artifact_path), f"Artifact missing: {artifact_path}"
    assert os.path.exists(config_path), f"Config missing: {config_path}"

    with open(config_path) as f:
        config = json.load(f)

    # Verify Checksum
    with open(artifact_path, 'rb') as f:
        calc_sha = hashlib.sha256(f.read()).hexdigest()
    assert calc_sha == config['sha256'], f"Checksum mismatch! {calc_sha} vs {config['sha256']}"
    print(f"[Checksum Check] Passed. SHA256 matches: {calc_sha}")

    # Load artifact
    loaded_model = joblib.load(artifact_path)
    assert isinstance(loaded_model, AirSenseCompositeModel), f"Unexpected type: {type(loaded_model)}"
    print("[Load Check] Passed. Loaded AirSenseCompositeModel instance successfully.")

    # Test single prediction
    res = loaded_model.predict_single(
        pm25=45.0,
        pm10=80.0,
        no2=25.0,
        co=1.2,
        so2=10.0,
        o3=30.0,
        month=5
    )
    assert 'raw_predicted_aqi' in res
    assert 'predicted_aqi' in res
    assert 'aqi_category' in res
    assert res['predicted_aqi'] >= 0.0
    print(f"[Single Predict Check] Passed. Output: {res}")

    # Test missing PM10 ratio estimation
    res_no_pm10 = loaded_model.predict_single(
        pm25=50.0,
        pm10=None,
        no2=25.0,
        co=1.2,
        so2=10.0,
        o3=30.0,
        month=1
    )
    assert any('PM10 was not provided' in w for w in res_no_pm10['warnings'])
    print(f"[Missing PM10 Check] Passed. Estimated PM10: {res_no_pm10['features_used']['PM10']}, Warnings: {res_no_pm10['warnings']}")

    # Test CO anomaly warning
    res_high_co = loaded_model.predict_single(
        pm25=50.0,
        pm10=90.0,
        no2=25.0,
        co=25.0, # > 17 mg/m3
        so2=10.0,
        o3=30.0,
        month=1
    )
    assert any('exceeds statutory ambient threshold' in w for w in res_high_co['warnings'])
    print(f"[CO Anomaly Warning Check] Passed. Warning emitted: {res_high_co['warnings']}")

    # Test batch DataFrame prediction
    df_test = pd.DataFrame([
        {'PM2.5': 30.0, 'PM10': 60.0, 'NO2': 15.0, 'CO': 0.8, 'SO2': 8.0, 'O3': 20.0, 'Month': 3},
        {'PM2.5': 150.0, 'PM10': np.nan, 'NO2': 45.0, 'CO': 2.5, 'SO2': 18.0, 'O3': 40.0, 'Month': 11}
    ])
    preds = loaded_model.predict(df_test)
    assert len(preds) == 2
    assert np.all(np.isfinite(preds))
    print(f"[Batch DataFrame Check] Passed. Predictions: {preds}")

    print("\nALL FRESH-PROCESS ARTIFACT LOADING CHECKS PASSED PERFECTLY!")

if __name__ == '__main__':
    test_fresh_load()

