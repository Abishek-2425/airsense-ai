import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
city_day['Date'] = pd.to_datetime(city_day['Date'])

print("="*70)
print("PART 4: PM10 MISSINGNESS AND IMPUTATION STRATEGY VERIFICATION")
print("="*70)

# Check missingness of PM10 and PM2.5 across all cities
summary = []
for city, grp in city_day.groupby('City'):
    n = len(grp)
    pm10_null = grp['PM10'].isnull().sum()
    pm25_null = grp['PM2.5'].isnull().sum()
    both_valid = (grp['PM10'].notnull() & grp['PM2.5'].notnull()).sum()
    summary.append({
        'City': city,
        'Total_Days': n,
        'PM10_Null': pm10_null,
        'PM10_Null_Pct': pm10_null / n * 100,
        'PM2.5_Null': pm25_null,
        'PM2.5_Null_Pct': pm25_null / n * 100,
        'Both_Valid': both_valid,
        'Both_Valid_Pct': both_valid / n * 100
    })

df_sum = pd.DataFrame(summary).sort_values('PM10_Null_Pct', ascending=False)
print("\nTop 10 Cities by PM10 Missingness:")
print(df_sum.head(10).to_string(index=False))

# Cities highlighted in audit: Lucknow, Patna, Chennai
print("\nAudit Target Cities:")
print(df_sum[df_sum['City'].isin(['Lucknow', 'Patna', 'Chennai'])].to_string(index=False))

# Calculate PM2.5 / PM10 ratio on paired observations (where PM10 > 0 and PM2.5 > 0)
paired = city_day[(city_day['PM10'] > 0) & (city_day['PM2.5'] > 0)].copy()
paired['Ratio_PM25_PM10'] = paired['PM2.5'] / paired['PM10']
paired['Ratio_PM10_PM25'] = paired['PM10'] / paired['PM2.5']

# Filter out unphysical ratios (e.g. PM2.5 > PM10 since PM2.5 is a physical subset of PM10)
# But let's check how often PM2.5 > PM10 occurs in real ambient data due to independent instrument errors!
unphysical = (paired['Ratio_PM25_PM10'] > 1.0).sum()
print(f"\nPaired PM2.5 and PM10 observations: {len(paired)}")
print(f"Observations where PM2.5 > PM10 (unphysical due to measurement noise): {unphysical} ({unphysical/len(paired)*100:.2f}%)")

valid_ratio = paired[(paired['Ratio_PM25_PM10'] <= 1.2)] # allow slight instrument noise tolerance

print("\nEmpirical PM2.5 / PM10 Ratio Statistics (National):")
print(valid_ratio['Ratio_PM25_PM10'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

# By City
city_ratios = valid_ratio.groupby('City')['Ratio_PM25_PM10'].agg(['count', 'median', 'mean', 'std'])
print("\nPM2.5 / PM10 Ratio by City (Top 10 cities by observation count):")
print(city_ratios.sort_values('count', ascending=False).head(10))

# By Season / Month
valid_ratio['Month'] = valid_ratio['Date'].dt.month
monthly_ratios = valid_ratio.groupby('Month')['Ratio_PM25_PM10'].agg(['count', 'median', 'mean'])
print("\nPM2.5 / PM10 Ratio by Month (National):")
print(monthly_ratios)

# Compare Imputation Strategies for predicting actual PM10:
# We evaluate on valid_ratio:
# Ground truth: PM10
# Strategy 1: Global Median of PM10 (the current production pipeline SimpleImputer)
# Strategy 2: National Median Ratio: PM10_pred = PM2.5 / median_ratio
# Strategy 3: City-specific Median Ratio (fallback to national)
# Strategy 4: Seasonal/Month Median Ratio

y_true = valid_ratio['PM10'].values
global_median = valid_ratio['PM10'].median()
pred_global_median = np.full_like(y_true, global_median)

national_ratio_median = valid_ratio['Ratio_PM25_PM10'].median()
pred_ratio_national = (valid_ratio['PM2.5'] / national_ratio_median).values

# City specific ratio
city_med_ratio = valid_ratio.groupby('City')['Ratio_PM25_PM10'].median()
valid_ratio['City_Ratio'] = valid_ratio['City'].map(city_med_ratio).fillna(national_ratio_median)
pred_ratio_city = (valid_ratio['PM2.5'] / valid_ratio['City_Ratio']).values

# Month specific ratio
month_med_ratio = valid_ratio.groupby('Month')['Ratio_PM25_PM10'].median()
valid_ratio['Month_Ratio'] = valid_ratio['Month'].map(month_med_ratio).fillna(national_ratio_median)
pred_ratio_month = (valid_ratio['PM2.5'] / valid_ratio['Month_Ratio']).values

def calc_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'Strategy': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

results = [
    calc_metrics(y_true, pred_global_median, "1. Global Median Imputation (Current Prod)"),
    calc_metrics(y_true, pred_ratio_national, "2. National PM2.5/PM10 Ratio Imputation"),
    calc_metrics(y_true, pred_ratio_month, "3. Monthly PM2.5/PM10 Ratio Imputation"),
    calc_metrics(y_true, pred_ratio_city, "4. City-Specific Ratio Imputation")
]

print("\n--- Imputation Strategy Comparison for PM10 Reconstruction ---")
print(pd.DataFrame(results).to_string(index=False))

