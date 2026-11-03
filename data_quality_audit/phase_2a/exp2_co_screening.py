import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load paired data exactly as defined
city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
stations = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'stations.csv'))
station_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'station_day.csv'))

city_day['Date'] = pd.to_datetime(city_day['Date'])
city_day = city_day.sort_values(['City', 'Date']).reset_index(drop=True)

city_day['Next_Date'] = city_day.groupby('City')['Date'].shift(-1)
city_day['Target_AQI'] = city_day.groupby('City')['AQI'].shift(-1)

mask_1day = (city_day['Next_Date'] - city_day['Date']) == pd.Timedelta(days=1)
valid_paired = city_day[mask_1day].dropna(subset=['Target_AQI']).copy()
valid_paired = valid_paired.sort_values('Date').reset_index(drop=True)

FEATURE_COLS = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
TARGET_COL = 'Target_AQI'

# Chronological partition
train_df = valid_paired[valid_paired['Date'] <= pd.to_datetime('2019-12-04')].copy().reset_index(drop=True)
embargo_df = valid_paired[valid_paired['Date'] == pd.to_datetime('2019-12-05')].copy().reset_index(drop=True)
test_df = valid_paired[valid_paired['Date'] >= pd.to_datetime('2019-12-06')].copy().reset_index(drop=True)

# Matched test set (has valid current day AQI for clean comparisons)
matched_test = test_df[test_df['AQI'].notnull()].copy().reset_index(drop=True)

# 2. Record EVERY flagged observation (> 17 mg/m3) in the dataset
flagged_all = city_day[city_day['CO'] > 17.0].copy()
# Determine station
# If City has only 1 station, station is known. Ahmedabad -> GJ001.
station_map = stations.groupby('City')['StationId'].unique().to_dict()
def map_station(row):
    st_list = station_map.get(row['City'], [])
    if len(st_list) == 1:
        return st_list[0]
    return f"{len(st_list)} stations in city"

flagged_all['Station_Inferred'] = flagged_all.apply(map_station, axis=1)
flagged_out = flagged_all[['City', 'Date', 'Station_Inferred', 'CO', 'AQI', 'PM2.5', 'NO2']].sort_values(['CO'], ascending=False)
flagged_out.to_csv(os.path.join(RESULTS_DIR, 'flagged_co_observations.csv'), index=False)
print(f"Recorded {len(flagged_out)} observations with CO > 17 mg/m3 to flagged_co_observations.csv")

# 3. Define Strategies for Training
# Strategy A: Raw (No screening)
# Strategy B: Flagging (Add binary feature CO_Flag_gt17)
# Strategy C: Excluding flagged (>17 mg/m3) from training
# Strategy D: Capping CO at 17.0 mg/m3 in training
# Strategy E: Robust statistical threshold (Capping at 10.0 mg/m3, CPCB Moderate ceiling / 99th pct non-Ahm)

def train_and_eval_pipeline(X_tr, y_tr, X_te, te_df, strategy_name, features):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    pipe.fit(X_tr[features], y_tr)
    
    # Feature importances
    rf = pipe.named_steps['regressor']
    importances = dict(zip(features, rf.feature_importances_))
    
    # Predict on test
    preds = pipe.predict(X_te[features])
    y_true = te_df[TARGET_COL].values
    
    # Helper for metrics
    def get_metrics(sub_mask, label):
        if sub_mask.sum() == 0:
            return {'Subset': label, 'N': 0, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
        y_sub = y_true[sub_mask]
        p_sub = preds[sub_mask]
        return {
            'Strategy': strategy_name,
            'Subset': label,
            'N': int(sub_mask.sum()),
            'MAE': mean_absolute_error(y_sub, p_sub),
            'RMSE': np.sqrt(mean_squared_error(y_sub, p_sub)),
            'R2': r2_score(y_sub, p_sub)
        }
    
    mask_all = np.ones(len(te_df), dtype=bool)
    mask_non_ahm = (te_df['City'] != 'Ahmedabad').values
    mask_ahm = (te_df['City'] == 'Ahmedabad').values
    mask_high_co = (te_df['CO'] > 17.0).values
    mask_normal_co = (te_df['CO'] <= 17.0).values
    mask_severe_aqi = (te_df['AQI'] >= 401).values
    
    eval_records = [
        get_metrics(mask_all, "All Matched Test"),
        get_metrics(mask_non_ahm, "Non-Ahmedabad Cities (25 Cities)"),
        get_metrics(mask_ahm, "Ahmedabad Only (GJ001)"),
        get_metrics(mask_high_co, "Test CO > 17 mg/m3"),
        get_metrics(mask_normal_co, "Test CO <= 17 mg/m3"),
        get_metrics(mask_severe_aqi, "Test Current AQI >= 401 (Severe)")
    ]
    
    return pipe, importances, eval_records

# --- Strategy A: Original (Raw) ---
print("Running Strategy A: Original (No screening)...")
X_tr_A = train_df.copy()
y_tr_A = train_df[TARGET_COL].values
X_te_A = matched_test.copy()
pipe_A, imp_A, res_A = train_and_eval_pipeline(X_tr_A, y_tr_A, X_te_A, matched_test, "Strategy A (Raw Baseline)", FEATURE_COLS)

# --- Strategy B: Flagging (Add binary column) ---
print("Running Strategy B: Flagging CO > 17 (Binary indicator)...")
feat_B = FEATURE_COLS + ['CO_Flag_gt17']
X_tr_B = train_df.copy()
X_tr_B['CO_Flag_gt17'] = (X_tr_B['CO'] > 17.0).astype(float)
y_tr_B = train_df[TARGET_COL].values
X_te_B = matched_test.copy()
X_te_B['CO_Flag_gt17'] = (X_te_B['CO'] > 17.0).astype(float)
pipe_B, imp_B, res_B = train_and_eval_pipeline(X_tr_B, y_tr_B, X_te_B, matched_test, "Strategy B (Flagging Indicator)", feat_B)

# --- Strategy C: Excluding Flagged (>17 mg/m3) from Training ---
print("Running Strategy C: Excluding CO > 17 from Training Only...")
mask_train_C = (train_df['CO'] <= 17.0) | (train_df['CO'].isnull())
X_tr_C = train_df[mask_train_C].copy().reset_index(drop=True)
y_tr_C = X_tr_C[TARGET_COL].values
X_te_C = matched_test.copy() # test set is completely intact with raw observations
pipe_C, imp_C, res_C = train_and_eval_pipeline(X_tr_C, y_tr_C, X_te_C, matched_test, "Strategy C (Exclusion from Train)", FEATURE_COLS)

# --- Strategy D: Capping CO at 17.0 mg/m3 in Training ---
print("Running Strategy D: Capping CO at 17.0 mg/m3 in Training...")
X_tr_D = train_df.copy()
X_tr_D['CO'] = X_tr_D['CO'].clip(upper=17.0)
y_tr_D = train_df[TARGET_COL].values
# In test, evaluate with raw inputs to see how model generalizes when fed real inputs
X_te_D = matched_test.copy()
pipe_D, imp_D, res_D = train_and_eval_pipeline(X_tr_D, y_tr_D, X_te_D, matched_test, "Strategy D (Capping Train at 17)", FEATURE_COLS)

# --- Strategy E: Robust Statistical Threshold (Capping at 10.0 mg/m3) ---
print("Running Strategy E: Robust Statistical Threshold (Capping at 10.0 mg/m3)...")
X_tr_E = train_df.copy()
X_tr_E['CO'] = X_tr_E['CO'].clip(upper=10.0)
y_tr_E = train_df[TARGET_COL].values
X_te_E = matched_test.copy()
pipe_E, imp_E, res_E = train_and_eval_pipeline(X_tr_E, y_tr_E, X_te_E, matched_test, "Strategy E (Robust Cap at 10)", FEATURE_COLS)

# Assemble Results
all_exp2_metrics = pd.DataFrame(res_A + res_B + res_C + res_D + res_E)
all_exp2_metrics.to_csv(os.path.join(RESULTS_DIR, 'exp2_co_screening_metrics.csv'), index=False)

# Feature Importances Table
imp_dict = {
    'Strategy A (Raw Baseline)': imp_A,
    'Strategy B (Flagging Indicator)': imp_B,
    'Strategy C (Exclusion from Train)': imp_C,
    'Strategy D (Capping Train at 17)': imp_D,
    'Strategy E (Robust Cap at 10)': imp_E
}
imp_df = pd.DataFrame(imp_dict)
imp_df.to_csv(os.path.join(RESULTS_DIR, 'exp2_feature_importances.csv'))

print("\n" + "="*80)
print("EXPERIMENT 2: CO ANOMALY SCREENING EVALUATION ON NON-AHMEDABAD CITIES (N=4703)")
print("="*80)
print(all_exp2_metrics[all_exp2_metrics['Subset'] == "Non-Ahmedabad Cities (25 Cities)"][['Strategy', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))

print("\n" + "="*80)
print("EXPERIMENT 2: CO ANOMALY SCREENING EVALUATION ON AHMEDABAD ONLY (N=197)")
print("="*80)
print(all_exp2_metrics[all_exp2_metrics['Subset'] == "Ahmedabad Only (GJ001)"][['Strategy', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))

print("\n" + "="*80)
print("EXPERIMENT 2: FEATURE IMPORTANCES ACROSS STRATEGIES")
print("="*80)
print((imp_df * 100).round(2).to_string())

