import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 1. Load data and pair exactly as production
city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
city_day['Date'] = pd.to_datetime(city_day['Date'])
city_day = city_day.sort_values(['City', 'Date']).reset_index(drop=True)

city_day['Next_Date'] = city_day.groupby('City')['Date'].shift(-1)
city_day['Target_AQI'] = city_day.groupby('City')['AQI'].shift(-1)

mask_1day = (city_day['Next_Date'] - city_day['Date']) == pd.Timedelta(days=1)
valid_paired = city_day[mask_1day].dropna(subset=['Target_AQI']).copy()
valid_paired = valid_paired.sort_values('Date').reset_index(drop=True)

FEATURE_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
TARGET_COLUMN = 'Target_AQI'

# Production 80/20 split
split_idx = int(len(valid_paired) * 0.8)
train_orig = valid_paired.iloc[:split_idx].copy().reset_index(drop=True)
test_orig = valid_paired.iloc[split_idx:].copy().reset_index(drop=True)

# Clean split with 1-day embargo (candidate policy)
train_clean = valid_paired[valid_paired['Date'] <= pd.to_datetime('2019-12-04')].copy().reset_index(drop=True)
embargo_clean = valid_paired[valid_paired['Date'] == pd.to_datetime('2019-12-05')].copy().reset_index(drop=True)
test_clean = valid_paired[valid_paired['Date'] >= pd.to_datetime('2019-12-06')].copy().reset_index(drop=True)

# Load production model
prod_model = joblib.load(os.path.join(PROJECT_ROOT, 'aqi_model.pkl'))
rf_prod = prod_model.named_steps['regressor']

print("="*70)
print("PART 3: REPRODUCE MODEL METRICS AND EVALUATE DISTORTION")
print("="*70)

# Genuine feature importances
importances = dict(zip(FEATURE_COLUMNS, rf_prod.feature_importances_))
print("Production Model Genuine Feature Importances:")
for k, v in importances.items():
    print(f"  {k:<6}: {v:.6f} ({v*100:.2f}%)")

def evaluate_set(pipe, df_eval, label=""):
    y_true = df_eval[TARGET_COLUMN].values
    X_eval = df_eval[FEATURE_COLUMNS]
    y_pred = pipe.predict(X_eval)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'label': label, 'N': len(df_eval), 'r2': r2, 'rmse': rmse, 'mae': mae}

# Evaluate production model on:
# 1. Original test set (matched N=4902)
matched_orig = test_orig[test_orig['AQI'].notnull()]
print("\n--- Production Model on Original Test Split ---")
print(evaluate_set(prod_model, test_orig, "Prod Model - All Original Test (N=4970)"))
print(evaluate_set(prod_model, matched_orig, "Prod Model - Matched Original Test (N=4902)"))

# 2. Clean test set (Date >= 2019-12-06)
matched_clean = test_clean[test_clean['AQI'].notnull()]
print("\n--- Production Model on Clean Test Split (Embargoed) ---")
print(evaluate_set(prod_model, test_clean, "Prod Model - All Clean Test (N=4968)"))
print(evaluate_set(prod_model, matched_clean, "Prod Model - Matched Clean Test (N=4900)"))

# 3. Clean test: Ahmedabad vs Non-Ahmedabad
clean_ahm = test_clean[test_clean['City'] == 'Ahmedabad']
clean_non_ahm = test_clean[test_clean['City'] != 'Ahmedabad']

clean_ahm_m = clean_ahm[clean_ahm['AQI'].notnull()]
clean_non_ahm_m = clean_non_ahm[clean_non_ahm['AQI'].notnull()]

print("\n--- Ahmedabad vs Non-Ahmedabad Performance on Clean Test ---")
print(evaluate_set(prod_model, clean_ahm, "Ahmedabad All (N=201)"))
print(evaluate_set(prod_model, clean_ahm_m, "Ahmedabad Matched (N=197)"))
print(evaluate_set(prod_model, clean_non_ahm, "Non-Ahmedabad All (N=4767)"))
print(evaluate_set(prod_model, clean_non_ahm_m, "Non-Ahmedabad Matched (N=4703)"))

# 4. Error on extreme CO vs normal CO
co_extreme_test = test_clean[test_clean['CO'] > 17]
co_normal_test = test_clean[test_clean['CO'] <= 17]

print("\n--- Error on Extreme CO (>17 mg/m3) vs Normal CO (<=17 mg/m3) in Test Set ---")
print(evaluate_set(prod_model, co_extreme_test, "Extreme CO (>17 mg/m3)"))
print(evaluate_set(prod_model, co_normal_test, "Normal CO (<=17 mg/m3)"))

# 5. ISOLATED EXPERIMENT: Train model WITHOUT Ahmedabad on training set
print("\n--- ISOLATED EXPERIMENT: Fit Pipeline WITHOUT Ahmedabad on Training Set ---")
train_no_ahm = train_clean[train_clean['City'] != 'Ahmedabad'].copy().reset_index(drop=True)

exp_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])

exp_pipe.fit(train_no_ahm[FEATURE_COLUMNS], train_no_ahm[TARGET_COLUMN].values)
rf_exp = exp_pipe.named_steps['regressor']
exp_importances = dict(zip(FEATURE_COLUMNS, rf_exp.feature_importances_))

print("\nFeature Importances WITHOUT Ahmedabad in Training:")
for k, v in exp_importances.items():
    diff = v - importances[k]
    print(f"  {k:<6}: {v:.6f} ({v*100:.2f}%)  [Delta: {diff*100:+.2f}%]")

print("\nPerformance of Experimental Model (Trained WITHOUT Ahmedabad):")
print(evaluate_set(exp_pipe, clean_non_ahm_m, "Exp Model on Non-Ahmedabad Matched Test (N=4703)"))
print(evaluate_set(prod_model, clean_non_ahm_m, "Prod Model on Non-Ahmedabad Matched Test (N=4703)"))
print(evaluate_set(exp_pipe, clean_ahm_m, "Exp Model on Ahmedabad Matched Test (N=197)"))
print(evaluate_set(prod_model, clean_ahm_m, "Prod Model on Ahmedabad Matched Test (N=197)"))
print(evaluate_set(exp_pipe, matched_clean, "Exp Model on All Matched Test (N=4900)"))
print(evaluate_set(prod_model, matched_clean, "Prod Model on All Matched Test (N=4900)"))

