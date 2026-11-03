import os
import sys
import json
import hashlib
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHASE_2B_DIR = os.path.join(PROJECT_ROOT, 'model_validation', 'phase_2b')
RC_DIR = os.path.join(PHASE_2B_DIR, 'release_candidate')
os.makedirs(RC_DIR, exist_ok=True)

# Add Phase 2B to sys.path so preprocessor can be imported cleanly
sys.path.insert(0, PHASE_2B_DIR)
from preprocessor import AirSensePreprocessor, AirSenseCompositeModel, FEATURE_ORDER, CO_ANOMALY_THRESHOLD, AQI_MAX_STATUTORY

print("="*80)
print("AIRSENSE AI — PHASE 2B: MODEL RETRAINING & EVALUATION")
print("="*80)

# 1. Load Data
city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
city_day['Date'] = pd.to_datetime(city_day['Date'])
city_day['Month'] = city_day['Date'].dt.month

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8]: return 'Monsoon'
    else: return 'Post-Monsoon'

city_day['Season'] = city_day['Month'].apply(get_season)
city_day = city_day.sort_values(['City', 'Date']).reset_index(drop=True)

# Create 1-day shifted target
city_day['Next_Date'] = city_day.groupby('City')['Date'].shift(-1)
city_day['Target_AQI'] = city_day.groupby('City')['AQI'].shift(-1)

mask_1day = (city_day['Next_Date'] - city_day['Date']) == pd.Timedelta(days=1)
valid_paired = city_day[mask_1day].dropna(subset=['Target_AQI']).copy()
valid_paired = valid_paired.sort_values('Date').reset_index(drop=True)

# Partition Boundaries
TRAIN_END = '2019-12-04'
EMBARGO_DATE = '2019-12-05'
TEST_START = '2019-12-06'

train_df = valid_paired[valid_paired['Date'] <= pd.to_datetime(TRAIN_END)].copy().reset_index(drop=True)
embargo_df = valid_paired[valid_paired['Date'] == pd.to_datetime(EMBARGO_DATE)].copy().reset_index(drop=True)
test_df = valid_paired[valid_paired['Date'] >= pd.to_datetime(TEST_START)].copy().reset_index(drop=True)
matched_test = test_df[test_df['AQI'].notnull()].copy().reset_index(drop=True)

y_test_raw = matched_test['Target_AQI'].values
y_test_capped = np.clip(y_test_raw, 0, AQI_MAX_STATUTORY)

print(f"Historical Train samples (<= {TRAIN_END}): {len(train_df)}")
print(f"Embargo samples ({EMBARGO_DATE}): {len(embargo_df)} (Excluded from training and testing)")
print(f"Held-Out Matched Test samples (>= {TEST_START}): {len(matched_test)}")

# 2. Fit Candidate Preprocessor strictly on historical train
preprocessor = AirSensePreprocessor()
preprocessor.fit(train_df[['PM2.5', 'PM10', 'Month', 'NO2', 'CO', 'SO2', 'O3']])
print(f"Fitted National PM2.5/PM10 Ratio: {preprocessor.national_ratio_:.4f}")
print("Fitted Monthly Ratios:")
for m, r in preprocessor.month_ratios_.items():
    print(f"  Month {m:02d}: {r:.4f}")

# 3. Model Training & Comparison Configurations

# Model A: Unchanged Production Model (aqi_model.pkl)
prod_model_path = os.path.join(PROJECT_ROOT, 'aqi_model.pkl')
prod_model = joblib.load(prod_model_path)

# Model B: Clean Baseline (SimpleImputer + RandomForest on clean split)
pipe_clean_base = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_clean_base.fit(train_df[FEATURE_ORDER], train_df['Target_AQI'].values)

# Model C: Monthly PM10 Ratio Model Alone
train_df_c = preprocessor.transform(train_df)
test_df_c = preprocessor.transform(matched_test)
pipe_ratio_only = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_ratio_only.fit(train_df_c[FEATURE_ORDER], train_df['Target_AQI'].values)

# Model D: CO-Screened Model Alone (Drop train CO > 17 mg/m3)
mask_train_d = (train_df['CO'] <= CO_ANOMALY_THRESHOLD) | (train_df['CO'].isnull())
train_df_d = train_df[mask_train_d].copy().reset_index(drop=True)
pipe_co_screen = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_co_screen.fit(train_df_d[FEATURE_ORDER], train_df_d['Target_AQI'].values)

# Model E: Target-Capped Model Alone (Target <= 500)
pipe_target_cap = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_target_cap.fit(train_df[FEATURE_ORDER], np.clip(train_df['Target_AQI'].values, 0, AQI_MAX_STATUTORY))

# Model F: Integrated Release Candidate (Phase 2B Candidate)
# - Monthly PM2.5/PM10 ratio imputation
# - Screen out training CO > 17 mg/m3
# - Cap training target at 500
train_df_f = preprocessor.transform(train_df)
train_df_f['Target_AQI'] = train_df['Target_AQI'].values
train_df_f['CO_Raw'] = train_df['CO'].values # for filtering check

mask_train_f = (train_df_f['CO_Raw'] <= CO_ANOMALY_THRESHOLD) | (train_df_f['CO_Raw'].isnull())
train_df_f_clean = train_df_f[mask_train_f].copy().reset_index(drop=True)
y_train_f = np.clip(train_df_f_clean['Target_AQI'].values, 0, AQI_MAX_STATUTORY)

regressor_rc = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
regressor_rc.fit(train_df_f_clean[FEATURE_ORDER], y_train_f)

rc_composite = AirSenseCompositeModel(preprocessor, regressor_rc, version="2.0.0")

# 4. Comprehensive Evaluation on Held-Out Test Set
models_dict = {
    'Persistence Baseline': {'preds_raw': matched_test['AQI'].values, 'preds_disp': np.clip(matched_test['AQI'].values, 0, AQI_MAX_STATUTORY)},
    'Production Model (aqi_model.pkl)': {'preds_raw': prod_model.predict(matched_test[FEATURE_ORDER]), 'preds_disp': np.clip(prod_model.predict(matched_test[FEATURE_ORDER]), 0, AQI_MAX_STATUTORY)},
    'Model B: Clean Baseline': {'preds_raw': pipe_clean_base.predict(matched_test[FEATURE_ORDER]), 'preds_disp': np.clip(pipe_clean_base.predict(matched_test[FEATURE_ORDER]), 0, AQI_MAX_STATUTORY)},
    'Model C: Monthly Ratio Only': {'preds_raw': pipe_ratio_only.predict(test_df_c[FEATURE_ORDER]), 'preds_disp': np.clip(pipe_ratio_only.predict(test_df_c[FEATURE_ORDER]), 0, AQI_MAX_STATUTORY)},
    'Model D: CO-Screened Only': {'preds_raw': pipe_co_screen.predict(matched_test[FEATURE_ORDER]), 'preds_disp': np.clip(pipe_co_screen.predict(matched_test[FEATURE_ORDER]), 0, AQI_MAX_STATUTORY)},
    'Model E: Target-Capped Only': {'preds_raw': pipe_target_cap.predict(matched_test[FEATURE_ORDER]), 'preds_disp': np.clip(pipe_target_cap.predict(matched_test[FEATURE_ORDER]), 0, AQI_MAX_STATUTORY)},
    'Model F: Release Candidate v2.0': {'preds_raw': rc_composite.predict(matched_test), 'preds_disp': np.clip(rc_composite.predict(matched_test), 0, AQI_MAX_STATUTORY)}
}

# Subsets for detailed reporting
subsets = {
    'All Matched Test': np.ones(len(matched_test), dtype=bool),
    'Non-Ahmedabad Cities (25 Cities)': (matched_test['City'] != 'Ahmedabad').values,
    'Ahmedabad Only (GJ001)': (matched_test['City'] == 'Ahmedabad').values,
    'Delhi': (matched_test['City'] == 'Delhi').values,
    'Lucknow': (matched_test['City'] == 'Lucknow').values,
    'Bengaluru': (matched_test['City'] == 'Bengaluru').values,
    'Chennai': (matched_test['City'] == 'Chennai').values,
    'Mumbai': (matched_test['City'] == 'Mumbai').values,
    'Patna': (matched_test['City'] == 'Patna').values,
    'Winter Season': (matched_test['Season'] == 'Winter').values,
    'Summer Season': (matched_test['Season'] == 'Summer').values,
    'Monsoon Season': (matched_test['Season'] == 'Monsoon').values,
    'Severe Episodes (Current AQI >= 401)': (matched_test['AQI'] >= 401).values,
    'Extreme CO (Test CO > 17 mg/m3)': (matched_test['CO'] > CO_ANOMALY_THRESHOLD).values
}

results_rows = []
for sub_name, mask in subsets.items():
    if mask.sum() == 0: continue
    for mode, y_true in [('raw_target', y_test_raw), ('operational_target_500', y_test_capped)]:
        yt = y_true[mask]
        for m_name, m_data in models_dict.items():
            yp = m_data['preds_raw'][mask] if mode == 'raw_target' else m_data['preds_disp'][mask]
            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            r2 = r2_score(yt, yp)
            bias = np.mean(yp - yt)
            results_rows.append({
                'Evaluation_Target': mode,
                'Subset': sub_name,
                'Model': m_name,
                'N': int(mask.sum()),
                'MAE': round(float(mae), 4),
                'RMSE': round(float(rmse), 4),
                'R2': round(float(r2), 4),
                'Mean_Bias': round(float(bias), 4)
            })

all_metrics_df = pd.DataFrame(results_rows)

# Save Comparison CSVs
all_metrics_df.to_csv(os.path.join(RC_DIR, 'model_performance_comparison.csv'), index=False)

city_metrics = all_metrics_df[all_metrics_df['Subset'].isin(['Delhi', 'Lucknow', 'Bengaluru', 'Chennai', 'Mumbai', 'Patna', 'Ahmedabad Only (GJ001)', 'Non-Ahmedabad Cities (25 Cities)'])]
city_metrics.to_csv(os.path.join(RC_DIR, 'city_level_metrics.csv'), index=False)

season_metrics = all_metrics_df[all_metrics_df['Subset'].isin(['Winter Season', 'Summer Season', 'Monsoon Season'])]
season_metrics.to_csv(os.path.join(RC_DIR, 'seasonal_metrics.csv'), index=False)

# 5. Missing-PM10 Inference Simulation
# What happens when PM10 is omitted in test requests?
missing_pm10_test = matched_test.copy()
missing_pm10_test['PM10'] = np.nan # drop PM10 completely to test ratio imputation!

preds_rc_no_pm10 = rc_composite.predict(missing_pm10_test)
preds_prod_no_pm10 = prod_model.predict(missing_pm10_test[FEATURE_ORDER])

sim_results = [
    {
        'Scenario': 'All Test: True PM10 Provided',
        'Model': 'Production Model (aqi_model.pkl)',
        'MAE': mean_absolute_error(y_test_capped, np.clip(models_dict['Production Model (aqi_model.pkl)']['preds_raw'], 0, 500)),
        'R2': r2_score(y_test_capped, np.clip(models_dict['Production Model (aqi_model.pkl)']['preds_raw'], 0, 500))
    },
    {
        'Scenario': 'All Test: True PM10 Provided',
        'Model': 'Release Candidate v2.0',
        'MAE': mean_absolute_error(y_test_capped, np.clip(models_dict['Model F: Release Candidate v2.0']['preds_raw'], 0, 500)),
        'R2': r2_score(y_test_capped, np.clip(models_dict['Model F: Release Candidate v2.0']['preds_raw'], 0, 500))
    },
    {
        'Scenario': 'All Test: PM10 Completely Missing (Simulated)',
        'Model': 'Production Model (Median Imputed)',
        'MAE': mean_absolute_error(y_test_capped, np.clip(preds_prod_no_pm10, 0, 500)),
        'R2': r2_score(y_test_capped, np.clip(preds_prod_no_pm10, 0, 500))
    },
    {
        'Scenario': 'All Test: PM10 Completely Missing (Simulated)',
        'Model': 'Release Candidate v2.0 (Ratio Imputed)',
        'MAE': mean_absolute_error(y_test_capped, np.clip(preds_rc_no_pm10, 0, 500)),
        'R2': r2_score(y_test_capped, np.clip(preds_rc_no_pm10, 0, 500))
    }
]
pd.DataFrame(sim_results).to_csv(os.path.join(RC_DIR, 'missing_pm10_scenario_metrics.csv'), index=False)

# 6. Feature Importances
imp_df = pd.DataFrame({
    'Production Model': prod_model.named_steps['regressor'].feature_importances_,
    'Model B (Clean Base)': pipe_clean_base.named_steps['regressor'].feature_importances_,
    'Model C (Ratio Only)': pipe_ratio_only.named_steps['regressor'].feature_importances_,
    'Model D (CO Screen)': pipe_co_screen.named_steps['regressor'].feature_importances_,
    'Model E (Target Cap)': pipe_target_cap.named_steps['regressor'].feature_importances_,
    'Model F (Release Candidate v2.0)': regressor_rc.feature_importances_
}, index=FEATURE_ORDER)
imp_df.to_csv(os.path.join(RC_DIR, 'feature_importance.csv'))

# 7. Serialize Artifacts
artifact_path = os.path.join(RC_DIR, 'candidate_pipeline.joblib')
joblib.dump(rc_composite, artifact_path)

# Compute Checksum
with open(artifact_path, 'rb') as f:
    sha256_hash = hashlib.sha256(f.read()).hexdigest()

# Training Config
training_config = {
    'model_name': 'AirSense AI Release Candidate v2.0',
    'version': '2.0.0',
    'creation_date': datetime.now().isoformat(),
    'train_date_end': TRAIN_END,
    'embargo_date': EMBARGO_DATE,
    'test_date_start': TEST_START,
    'train_samples': len(train_df_f_clean),
    'test_samples': len(matched_test),
    'features': FEATURE_ORDER,
    'hyperparameters': {
        'n_estimators': 100,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'preprocessing': {
        'pm10_imputation': 'Monthly physical PM2.5/PM10 median ratio',
        'co_screening': f'Exclusion of training records with CO > {CO_ANOMALY_THRESHOLD} mg/m3',
        'target_capping': f'Training target capped at {AQI_MAX_STATUTORY}'
    },
    'sha256': sha256_hash
}
with open(os.path.join(RC_DIR, 'training_config.json'), 'w') as f:
    json.dump(training_config, f, indent=2)

# Preprocessor Metadata
prep_meta = {
    'national_ratio': preprocessor.national_ratio_,
    'month_ratios': preprocessor.month_ratios_,
    'global_medians': preprocessor.global_medians_,
    'co_anomaly_threshold': CO_ANOMALY_THRESHOLD,
    'aqi_max_statutory': AQI_MAX_STATUTORY
}
with open(os.path.join(RC_DIR, 'preprocessor_metadata.json'), 'w') as f:
    json.dump(prep_meta, f, indent=2)

# Model Metadata for API exposure
model_metadata = {
    'model_name': 'AirSense AI Release Candidate v2.0',
    'model_version': '2.0.0',
    'status': 'Release Candidate',
    'trained_timestamp': datetime.now().isoformat(),
    'sha256': sha256_hash,
    'training_data': {
        'source': 'Central Pollution Control Board (CPCB) India Continuous Ambient Air Quality Data',
        'train_window': f'2015-01-01 to {TRAIN_END}',
        'embargo_date': EMBARGO_DATE,
        'evaluation_window': f'{TEST_START} to 2020-06-30',
        'usable_train_pairs': len(train_df_f_clean),
        'usable_test_pairs': len(matched_test)
    },
    'feature_schema': {
        'expected_order': FEATURE_ORDER,
        'units': {
            'PM2.5': 'ug/m3',
            'PM10': 'ug/m3',
            'NO2': 'ug/m3',
            'CO': 'mg/m3',
            'SO2': 'ug/m3',
            'O3': 'ug/m3'
        },
        'feature_importances': {k: round(float(v), 6) for k, v in zip(FEATURE_ORDER, regressor_rc.feature_importances_)}
    },
    'evaluation_metrics': {
        'operational_scale_cpcb_500': {
            'all_cities_mae': float(all_metrics_df[(all_metrics_df['Evaluation_Target'] == 'operational_target_500') & (all_metrics_df['Subset'] == 'All Matched Test') & (all_metrics_df['Model'] == 'Model F: Release Candidate v2.0')]['MAE'].values[0]),
            'all_cities_rmse': float(all_metrics_df[(all_metrics_df['Evaluation_Target'] == 'operational_target_500') & (all_metrics_df['Subset'] == 'All Matched Test') & (all_metrics_df['Model'] == 'Model F: Release Candidate v2.0')]['RMSE'].values[0]),
            'all_cities_r2': float(all_metrics_df[(all_metrics_df['Evaluation_Target'] == 'operational_target_500') & (all_metrics_df['Subset'] == 'All Matched Test') & (all_metrics_df['Model'] == 'Model F: Release Candidate v2.0')]['R2'].values[0]),
            'non_ahmedabad_mae': float(all_metrics_df[(all_metrics_df['Evaluation_Target'] == 'operational_target_500') & (all_metrics_df['Subset'] == 'Non-Ahmedabad Cities (25 Cities)') & (all_metrics_df['Model'] == 'Model F: Release Candidate v2.0')]['MAE'].values[0]),
            'non_ahmedabad_r2': float(all_metrics_df[(all_metrics_df['Evaluation_Target'] == 'operational_target_500') & (all_metrics_df['Subset'] == 'Non-Ahmedabad Cities (25 Cities)') & (all_metrics_df['Model'] == 'Model F: Release Candidate v2.0')]['R2'].values[0]),
            'delhi_r2': float(all_metrics_df[(all_metrics_df['Evaluation_Target'] == 'operational_target_500') & (all_metrics_df['Subset'] == 'Delhi') & (all_metrics_df['Model'] == 'Model F: Release Candidate v2.0')]['R2'].values[0])
        }
    }
}
with open(os.path.join(RC_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("\n" + "="*80)
print(f"RELEASE CANDIDATE SERIALIZED SUCCESSFULLY: {artifact_path}")
print(f"SHA256: {sha256_hash}")
print("="*80)
