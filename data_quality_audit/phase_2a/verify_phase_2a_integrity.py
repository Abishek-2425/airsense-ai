import os
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_2A = os.path.join(PROJECT_ROOT, 'data_quality_audit', 'phase_2a', 'results')

print("="*80)
print("PHASE 2A FINAL INDEPENDENT VERIFICATION")
print("="*80)

# 1. Verify PM10 Imputation
pm10_ov = pd.read_csv(os.path.join(RESULTS_2A, 'exp1_pm10_imputation_overall.csv'))
print("\n--- PM10 Imputation Overall Metrics ---")
print(pm10_ov[['Fold', 'Strategy', 'N', 'MAE', 'RMSE', 'R2']])

# 2. Verify CO Screening
co_metrics = pd.read_csv(os.path.join(RESULTS_2A, 'exp2_co_screening_metrics.csv'))
print("\n--- CO Screening Metrics ---")
print(co_metrics[co_metrics['Subset'].isin(['All Matched Test', 'Non-Ahmedabad Cities (25 Cities)', 'Ahmedabad Only (GJ001)'])][['Strategy', 'Subset', 'N', 'MAE', 'RMSE', 'R2']])

co_imp = pd.read_csv(os.path.join(RESULTS_2A, 'exp2_feature_importances.csv'), index_col=0)
print("\n--- CO Feature Importances (%) ---")
print((co_imp * 100).round(2))

# 3. Verify AQI Above 500
aqi_metrics = pd.read_csv(os.path.join(RESULTS_2A, 'exp3_aqi_above_500_metrics.csv'))
print("\n--- Target Capping Metrics ---")
print(aqi_metrics[aqi_metrics['Subset'].isin(['All Matched Test', 'Non-Ahmedabad Cities (25 Cities)', 'Severe Episode (Current AQI >= 401)'])][['Model', 'Evaluation_Ground_Truth', 'Subset', 'N', 'MAE', 'RMSE', 'R2']])

# 4. Verify Model Comparison
model_comp = pd.read_csv(os.path.join(RESULTS_2A, 'exp4_comprehensive_model_comparison.csv'))
print("\n--- Model Comparison (Raw GT, All Matched Test) ---")
print(model_comp[(model_comp['Ground_Truth_Mode'] == 'raw') & (model_comp['Subset'] == 'All Matched Test')][['Model', 'N', 'MAE', 'RMSE', 'R2', 'Mean_Bias']])

print("\n--- Model Comparison (Capped GT 500, All Matched Test) ---")
print(model_comp[(model_comp['Ground_Truth_Mode'] == 'capped') & (model_comp['Subset'] == 'All Matched Test')][['Model', 'N', 'MAE', 'RMSE', 'R2', 'Mean_Bias']])

print("\n--- Model Comparison (Non-Ahmedabad, N=4703) ---")
print(model_comp[(model_comp['Ground_Truth_Mode'] == 'raw') & (model_comp['Subset'] == 'Non-Ahmedabad Cities (25 Cities)')][['Model', 'N', 'MAE', 'RMSE', 'R2']])

pipe_imp = pd.read_csv(os.path.join(RESULTS_2A, 'exp4_pipeline_feature_importances.csv'), index_col=0)
print("\n--- Pipeline Feature Importances (%) ---")
print((pipe_imp * 100).round(2))

