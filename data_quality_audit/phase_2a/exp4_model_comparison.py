import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# Target shift
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
matched_test = test_df[test_df['AQI'].notnull()].copy().reset_index(drop=True)

y_test_raw = matched_test[TARGET_COL].values
y_test_capped = np.clip(y_test_raw, 0, 500)

print(f"Train samples: {len(train_df)}, Embargo samples: {len(embargo_df)}, Matched Test samples: {len(matched_test)}")

# Custom Scikit-learn Transformer for Monthly PM2.5/PM10 Ratio Imputation
class MonthlyRatioPM10Imputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing PM10 using monthly PM2.5/PM10 median ratio derived strictly from training data.
    Falls back to national ratio, then global median PM10.
    """
    def __init__(self):
        self.month_ratios_ = {}
        self.national_ratio_ = 0.47
        self.global_median_pm10_ = 101.88

    def fit(self, X, y=None):
        # Expects X to have columns: PM2.5, PM10, Month
        paired = X[(X['PM10'] > 0) & (X['PM2.5'] > 0)].copy()
        paired = paired[paired['PM2.5'] <= 1.2 * paired['PM10']]
        paired['ratio'] = paired['PM2.5'] / paired['PM10']
        
        self.national_ratio_ = float(paired['ratio'].median())
        self.month_ratios_ = paired.groupby('Month')['ratio'].median().to_dict()
        self.global_median_pm10_ = float(X['PM10'].median())
        return self

    def transform(self, X):
        X_out = X.copy()
        # If PM10 is missing, estimate from PM2.5
        missing_pm10 = X_out['PM10'].isnull()
        for idx in X_out[missing_pm10].index:
            pm25 = X_out.loc[idx, 'PM2.5']
            month = X_out.loc[idx, 'Month']
            if pd.notnull(pm25) and pm25 > 0:
                r = self.month_ratios_.get(month, self.national_ratio_)
                X_out.loc[idx, 'PM10'] = pm25 / r
            else:
                X_out.loc[idx, 'PM10'] = self.global_median_pm10_
        return X_out

# 2. Fit and Evaluate All Candidate Pipelines

# Baseline 1: Persistence Model
preds_persistence = matched_test['AQI'].values

# Baseline 2: Unchanged Production Model
prod_model_path = os.path.join(PROJECT_ROOT, 'aqi_model.pkl')
prod_model = joblib.load(prod_model_path)
preds_prod_model = prod_model.predict(matched_test[FEATURE_COLS])

# Pipeline 1: Clean Baseline (Retrained identically on train split)
pipe_1 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_1.fit(train_df[FEATURE_COLS], train_df[TARGET_COL].values)
preds_pipe_1 = pipe_1.predict(matched_test[FEATURE_COLS])

# Pipeline 2: Monthly Ratio Imputation Alone
ratio_imputer_p2 = MonthlyRatioPM10Imputer()
ratio_imputer_p2.fit(train_df[['PM2.5', 'PM10', 'Month']])

train_df_p2 = ratio_imputer_p2.transform(train_df[['PM2.5', 'PM10', 'Month'] + [c for c in FEATURE_COLS if c not in ['PM2.5', 'PM10']]])
test_df_p2 = ratio_imputer_p2.transform(matched_test[['PM2.5', 'PM10', 'Month'] + [c for c in FEATURE_COLS if c not in ['PM2.5', 'PM10']]])

pipe_2 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_2.fit(train_df_p2[FEATURE_COLS], train_df[TARGET_COL].values)
preds_pipe_2 = pipe_2.predict(test_df_p2[FEATURE_COLS])

# Pipeline 3: CO Anomaly Screening Alone (Exclude CO > 17 mg/m3 from training)
train_df_p3 = train_df[(train_df['CO'] <= 17.0) | (train_df['CO'].isnull())].copy().reset_index(drop=True)
pipe_3 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_3.fit(train_df_p3[FEATURE_COLS], train_df_p3[TARGET_COL].values)
preds_pipe_3 = pipe_3.predict(matched_test[FEATURE_COLS])

# Pipeline 4: Target Capping Alone (Cap training target at 500)
pipe_4 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_4.fit(train_df[FEATURE_COLS], np.clip(train_df[TARGET_COL].values, 0, 500))
preds_pipe_4 = pipe_4.predict(matched_test[FEATURE_COLS])

# Pipeline 5: Integrated Preprocessing Pipeline
# - Monthly Ratio Imputation for PM10
# - Exclude training CO > 17 mg/m3
# - Cap training target at 500
ratio_imputer_p5 = MonthlyRatioPM10Imputer()
ratio_imputer_p5.fit(train_df[['PM2.5', 'PM10', 'Month']])

train_df_p5_all = ratio_imputer_p5.transform(train_df[['PM2.5', 'PM10', 'Month'] + [c for c in FEATURE_COLS if c not in ['PM2.5', 'PM10']]])
train_df_p5_all['Target_AQI'] = train_df['Target_AQI'].values

# Filter CO > 17 mg/m3 from training
mask_p5_co = (train_df_p5_all['CO'] <= 17.0) | (train_df_p5_all['CO'].isnull())
train_df_p5_filtered = train_df_p5_all[mask_p5_co].copy().reset_index(drop=True)
y_train_p5 = np.clip(train_df_p5_filtered['Target_AQI'].values, 0, 500)

test_df_p5 = ratio_imputer_p5.transform(matched_test[['PM2.5', 'PM10', 'Month'] + [c for c in FEATURE_COLS if c not in ['PM2.5', 'PM10']]])

pipe_5 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
])
pipe_5.fit(train_df_p5_filtered[FEATURE_COLS], y_train_p5)
preds_pipe_5_raw = pipe_5.predict(test_df_p5[FEATURE_COLS])
preds_pipe_5_disp = np.clip(preds_pipe_5_raw, 0, 500)

# Model Dictionary
all_models = {
    'Persistence Baseline': {'preds': preds_persistence, 'clamped': np.clip(preds_persistence, 0, 500)},
    'Production Model (aqi_model.pkl)': {'preds': preds_prod_model, 'clamped': np.clip(preds_prod_model, 0, 500)},
    'Pipeline 1: Clean Baseline': {'preds': preds_pipe_1, 'clamped': np.clip(preds_pipe_1, 0, 500)},
    'Pipeline 2: Monthly Ratio Imputation': {'preds': preds_pipe_2, 'clamped': np.clip(preds_pipe_2, 0, 500)},
    'Pipeline 3: CO Anomaly Screening': {'preds': preds_pipe_3, 'clamped': np.clip(preds_pipe_3, 0, 500)},
    'Pipeline 4: Target Capping (500)': {'preds': preds_pipe_4, 'clamped': np.clip(preds_pipe_4, 0, 500)},
    'Pipeline 5: Integrated Pipeline': {'preds': preds_pipe_5_raw, 'clamped': preds_pipe_5_disp}
}

# 3. Comprehensive Evaluation
def evaluate_all_subsets(eval_ground_truth_mode='raw'):
    y_true = y_test_raw if eval_ground_truth_mode == 'raw' else y_test_capped
    results_list = []
    
    # Define Subsets
    subsets_dict = {
        'All Matched Test': np.ones(len(matched_test), dtype=bool),
        'Non-Ahmedabad Cities (25 Cities)': (matched_test['City'] != 'Ahmedabad').values,
        'Ahmedabad Only (GJ001)': (matched_test['City'] == 'Ahmedabad').values,
        'Winter Season': (matched_test['Season'] == 'Winter').values,
        'Summer Season': (matched_test['Season'] == 'Summer').values,
        'Monsoon Season': (matched_test['Season'] == 'Monsoon').values,
        'Post-Monsoon Season': (matched_test['Season'] == 'Post-Monsoon').values,
        'Severe Episodes (Current AQI >= 401)': (matched_test['AQI'] >= 401).values,
        'Delhi Only': (matched_test['City'] == 'Delhi').values,
        'Lucknow Only': (matched_test['City'] == 'Lucknow').values,
        'Bengaluru Only': (matched_test['City'] == 'Bengaluru').values,
        'Chennai Only': (matched_test['City'] == 'Chennai').values,
        'Mumbai Only': (matched_test['City'] == 'Mumbai').values,
        'Patna Only': (matched_test['City'] == 'Patna').values
    }
    
    for sub_name, mask in subsets_dict.items():
        if mask.sum() == 0: continue
        yt = y_true[mask]
        for m_name, m_data in all_models.items():
            yp = m_data['preds'][mask] if eval_ground_truth_mode == 'raw' else m_data['clamped'][mask]
            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            r2 = r2_score(yt, yp)
            bias = np.mean(yp - yt)
            results_list.append({
                'Ground_Truth_Mode': eval_ground_truth_mode,
                'Subset': sub_name,
                'Model': m_name,
                'N': int(mask.sum()),
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Mean_Bias': bias
            })
    return results_list

raw_evals = evaluate_all_subsets('raw')
capped_evals = evaluate_all_subsets('capped')

all_eval_df = pd.DataFrame(raw_evals + capped_evals)
all_eval_df.to_csv(os.path.join(RESULTS_DIR, 'exp4_comprehensive_model_comparison.csv'), index=False)

# Feature Importances for Retrained Pipelines
imp_table = {
    'Production Model': dict(zip(FEATURE_COLS, prod_model.named_steps['regressor'].feature_importances_)),
    'Pipeline 1 (Clean Baseline)': dict(zip(FEATURE_COLS, pipe_1.named_steps['regressor'].feature_importances_)),
    'Pipeline 2 (Ratio Imputation)': dict(zip(FEATURE_COLS, pipe_2.named_steps['regressor'].feature_importances_)),
    'Pipeline 3 (CO Screening)': dict(zip(FEATURE_COLS, pipe_3.named_steps['regressor'].feature_importances_)),
    'Pipeline 4 (Target Capping)': dict(zip(FEATURE_COLS, pipe_4.named_steps['regressor'].feature_importances_)),
    'Pipeline 5 (Integrated)': dict(zip(FEATURE_COLS, pipe_5.named_steps['regressor'].feature_importances_))
}
imp_summary_df = pd.DataFrame(imp_table)
imp_summary_df.to_csv(os.path.join(RESULTS_DIR, 'exp4_pipeline_feature_importances.csv'))

print("\n" + "="*80)
print("EXPERIMENT 4: OVERALL COMPARISON (RAW GROUND TRUTH, N=4900)")
print("="*80)
print(all_eval_df[(all_eval_df['Ground_Truth_Mode'] == 'raw') & (all_eval_df['Subset'] == 'All Matched Test')][['Model', 'N', 'MAE', 'RMSE', 'R2', 'Mean_Bias']].to_string(index=False))

print("\n" + "="*80)
print("EXPERIMENT 4: NON-AHMEDABAD CITIES COMPARISON (N=4703)")
print("="*80)
print(all_eval_df[(all_eval_df['Ground_Truth_Mode'] == 'raw') & (all_eval_df['Subset'] == 'Non-Ahmedabad Cities (25 Cities)')][['Model', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))

print("\n" + "="*80)
print("EXPERIMENT 4: OPERATIONAL COMPARISON (CAPPED GROUND TRUTH 500, N=4900)")
print("="*80)
print(all_eval_df[(all_eval_df['Ground_Truth_Mode'] == 'capped') & (all_eval_df['Subset'] == 'All Matched Test')][['Model', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))

print("\n" + "="*80)
print("EXPERIMENT 4: PIPELINE FEATURE IMPORTANCES (%)")
print("="*80)
print((imp_summary_df * 100).round(2).to_string())

