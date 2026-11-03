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

# 1. Load Data
city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
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
test_df = valid_paired[valid_paired['Date'] >= pd.to_datetime('2019-12-06')].copy().reset_index(drop=True)
matched_test = test_df[test_df['AQI'].notnull()].copy().reset_index(drop=True)

# Prepare targets
y_train_raw = train_df[TARGET_COL].values
y_train_capped = np.clip(y_train_raw, 0, 500)

y_test_raw = matched_test[TARGET_COL].values
y_test_capped = np.clip(y_test_raw, 0, 500)

print(f"Train samples: {len(train_df)}, Raw Max Target: {y_train_raw.max()}, Capped Max Target: {y_train_capped.max()}")
print(f"Test matched samples: {len(matched_test)}, Raw Max Target: {y_test_raw.max()}, Capped Max Target: {y_test_capped.max()}")

# Model 1: Trained on Raw Target
print("\nTraining Model 1 (Raw Target)...")
pipe_raw = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_raw.fit(train_df[FEATURE_COLS], y_train_raw)

# Model 2: Trained on Capped Target
print("Training Model 2 (Capped Target at 500)...")
pipe_capped = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_capped.fit(train_df[FEATURE_COLS], y_train_capped)

# Predictions
preds_raw_model_raw = pipe_raw.predict(matched_test[FEATURE_COLS])
preds_raw_model_disp = np.clip(preds_raw_model_raw, 0, 500)

preds_capped_model_raw = pipe_capped.predict(matched_test[FEATURE_COLS])
preds_capped_model_disp = np.clip(preds_capped_model_raw, 0, 500)

# Evaluate function
def evaluate_target_experiment(y_true, y_pred, model_name, eval_mode, subset_name, mask):
    if mask.sum() == 0:
        return {}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    bias = np.mean(yp - yt)
    return {
        'Model': model_name,
        'Evaluation_Ground_Truth': eval_mode,
        'Subset': subset_name,
        'N': int(mask.sum()),
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Mean_Bias': bias
    }

subsets = {
    'All Matched Test': np.ones(len(matched_test), dtype=bool),
    'Non-Ahmedabad Cities (25 Cities)': (matched_test['City'] != 'Ahmedabad').values,
    'Ahmedabad Only (GJ001)': (matched_test['City'] == 'Ahmedabad').values,
    'Target AQI > 500 Cases': (matched_test[TARGET_COL] > 500).values,
    'Severe Episode (Current AQI >= 401)': (matched_test['AQI'] >= 401).values
}

results = []
for s_name, s_mask in subsets.items():
    # 1. Raw Model evaluated against Raw Ground Truth
    results.append(evaluate_target_experiment(y_test_raw, preds_raw_model_raw, "Raw Target Model", "Raw Ground Truth", s_name, s_mask))
    # 2. Raw Model with Display Clamping evaluated against Capped Ground Truth
    results.append(evaluate_target_experiment(y_test_capped, preds_raw_model_disp, "Raw Target Model (Clamped Display)", "Capped Ground Truth (500)", s_name, s_mask))
    # 3. Capped Target Model evaluated against Raw Ground Truth
    results.append(evaluate_target_experiment(y_test_raw, preds_capped_model_raw, "Capped Target Model", "Raw Ground Truth", s_name, s_mask))
    # 4. Capped Target Model evaluated against Capped Ground Truth
    results.append(evaluate_target_experiment(y_test_capped, preds_capped_model_disp, "Capped Target Model (Operational)", "Capped Ground Truth (500)", s_name, s_mask))

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'exp3_aqi_above_500_metrics.csv'), index=False)

# Feature Importances
imp_raw = dict(zip(FEATURE_COLS, pipe_raw.named_steps['regressor'].feature_importances_))
imp_capped = dict(zip(FEATURE_COLS, pipe_capped.named_steps['regressor'].feature_importances_))

imp_comp = pd.DataFrame({
    'Raw Target Model': imp_raw,
    'Capped Target Model': imp_capped
})
imp_comp.to_csv(os.path.join(RESULTS_DIR, 'exp3_feature_importances.csv'))

print("\n" + "="*80)
print("EXPERIMENT 3: TARGET CAPPING EVALUATION ON HELD-OUT TEST")
print("="*80)
print(results_df[['Model', 'Evaluation_Ground_Truth', 'Subset', 'N', 'MAE', 'RMSE', 'R2', 'Mean_Bias']].to_string(index=False))

print("\n" + "="*80)
print("EXPERIMENT 3: FEATURE IMPORTANCE COMPARISON")
print("="*80)
print((imp_comp * 100).round(2).to_string())

