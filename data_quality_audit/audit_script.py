import os
import pandas as pd
import numpy as np

CSV_PATH = os.path.join("archive", "city_day.csv")
STATION_DAY_PATH = os.path.join("archive", "station_day.csv")

print("--- LOADING DATASET ---")
df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")

features_6 = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
all_pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
targets = ['AQI']

# 1. Feature Level Summary
summary_rows = []
for col in features_6 + targets + ['NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene']:
    s = df[col].dropna()
    n_miss = int(df[col].isnull().sum())
    pct_miss = (n_miss / len(df)) * 100.0
    n_valid = int(len(s))
    
    if n_valid > 0:
        c_min = float(s.min())
        c_max = float(s.max())
        c_mean = float(s.mean())
        c_median = float(s.median())
        c_std = float(s.std())
        p01 = float(s.quantile(0.01))
        p05 = float(s.quantile(0.05))
        p25 = float(s.quantile(0.25))
        p75 = float(s.quantile(0.75))
        p95 = float(s.quantile(0.95))
        p99 = float(s.quantile(0.99))
        p999 = float(s.quantile(0.999))
        
        iqr = p75 - p25
        lower_bound = p25 - 1.5 * iqr
        upper_bound = p75 + 1.5 * iqr
        
        outliers_low = int((s < lower_bound).sum())
        outliers_high = int((s > upper_bound).sum())
        total_outliers = outliers_low + outliers_high
        
        n_zeros = int((s == 0).sum())
        n_negatives = int((s < 0).sum())
    else:
        c_min = c_max = c_mean = c_median = c_std = np.nan
        p01 = p05 = p25 = p75 = p95 = p99 = p999 = np.nan
        lower_bound = upper_bound = iqr = np.nan
        outliers_low = outliers_high = total_outliers = 0
        n_zeros = n_negatives = 0

    summary_rows.append({
        'feature': col,
        'non_null_count': n_valid,
        'missing_count': n_miss,
        'missing_percentage': round(pct_miss, 2),
        'min': round(c_min, 4),
        'max': round(c_max, 4),
        'mean': round(c_mean, 4),
        'median': round(c_median, 4),
        'std': round(c_std, 4),
        'p01': round(p01, 4),
        'p05': round(p05, 4),
        'p25_q1': round(p25, 4),
        'p75_q3': round(p75, 4),
        'p95': round(p95, 4),
        'p99': round(p99, 4),
        'p99_9': round(p999, 4),
        'iqr': round(iqr, 4),
        'iqr_lower_bound': round(lower_bound, 4),
        'iqr_upper_bound': round(upper_bound, 4),
        'outlier_count_iqr': total_outliers,
        'outlier_percentage_iqr': round((total_outliers / n_valid) * 100, 2) if n_valid > 0 else 0,
        'zero_count': n_zeros,
        'negative_count': n_negatives
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join("data_quality_audit", "DATA_QUALITY_SUMMARY.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Saved {summary_csv_path}")

# 2. City Level Analysis
city_rows = []
for city in sorted(df['City'].unique()):
    cdf = df[df['City'] == city].sort_values('Date')
    n_rows = len(cdf)
    start_date = cdf['Date'].min().strftime('%Y-%m-%d')
    end_date = cdf['Date'].max().strftime('%Y-%m-%d')
    total_days = (cdf['Date'].max() - cdf['Date'].min()).days + 1
    missing_calendar_days = total_days - n_rows
    
    # Missing percentages
    pm25_miss = round((cdf['PM2.5'].isnull().sum() / n_rows) * 100, 2)
    pm10_miss = round((cdf['PM10'].isnull().sum() / n_rows) * 100, 2)
    no2_miss = round((cdf['NO2'].isnull().sum() / n_rows) * 100, 2)
    co_miss = round((cdf['CO'].isnull().sum() / n_rows) * 100, 2)
    so2_miss = round((cdf['SO2'].isnull().sum() / n_rows) * 100, 2)
    o3_miss = round((cdf['O3'].isnull().sum() / n_rows) * 100, 2)
    aqi_miss = round((cdf['AQI'].isnull().sum() / n_rows) * 100, 2)
    
    # Extreme values
    max_pm25 = round(float(cdf['PM2.5'].dropna().max()), 2) if cdf['PM2.5'].notnull().any() else np.nan
    max_pm10 = round(float(cdf['PM10'].dropna().max()), 2) if cdf['PM10'].notnull().any() else np.nan
    max_co = round(float(cdf['CO'].dropna().max()), 2) if cdf['CO'].notnull().any() else np.nan
    max_aqi = round(float(cdf['AQI'].dropna().max()), 2) if cdf['AQI'].notnull().any() else np.nan
    
    # Extreme CO observations (> 17 mg/m3 CPCB severe)
    co_extreme_count = int((cdf['CO'] > 17.0).sum())
    pm10_extreme_count = int((cdf['PM10'] >= 1000.0).sum())
    
    city_rows.append({
        'city': city,
        'records_count': n_rows,
        'start_date': start_date,
        'end_date': end_date,
        'calendar_day_span': total_days,
        'missing_calendar_days': missing_calendar_days,
        'pm25_missing_pct': pm25_miss,
        'pm10_missing_pct': pm10_miss,
        'no2_missing_pct': no2_miss,
        'co_missing_pct': co_miss,
        'so2_missing_pct': so2_miss,
        'o3_missing_pct': o3_miss,
        'aqi_missing_pct': aqi_miss,
        'max_pm25': max_pm25,
        'max_pm10': max_pm10,
        'max_co': max_co,
        'max_aqi': max_aqi,
        'co_gt_17_count': co_extreme_count,
        'pm10_ge_1000_count': pm10_extreme_count
    })

city_df = pd.DataFrame(city_rows)
city_csv_path = os.path.join("data_quality_audit", "CITY_DATA_QUALITY.csv")
city_df.to_csv(city_csv_path, index=False)
print(f"Saved {city_csv_path}")

# 3. Deep Dive into CO and Other Extremes
print("\n" + "="*60)
print("TOP 20 CO OBSERVATIONS IN city_day.csv")
print("="*60)
top_co = df.nlargest(20, 'CO')[['City', 'Date', 'CO', 'PM2.5', 'PM10', 'AQI']]
print(top_co.to_string())

print("\n" + "="*60)
print("TOP 10 PM2.5 OBSERVATIONS IN city_day.csv")
print("="*60)
top_pm25 = df.nlargest(10, 'PM2.5')[['City', 'Date', 'PM2.5', 'PM10', 'AQI']]
print(top_pm25.to_string())

print("\n" + "="*60)
print("PM10 OBSERVATIONS >= 900")
print("="*60)
top_pm10 = df[df['PM10'] >= 900][['City', 'Date', 'PM10', 'PM2.5', 'AQI']]
print(top_pm10.to_string())

print("\n" + "="*60)
print("AQI OBSERVATIONS > 500 (BEYOND NAQI SCALE)")
print("="*60)
high_aqi = df[df['AQI'] > 500][['City', 'Date', 'AQI', 'AQI_Bucket', 'PM2.5', 'PM10']]
print(f"Total rows with AQI > 500: {len(high_aqi)}")
print(high_aqi.head(15).to_string())

print("\n" + "="*60)
print("AQI_BUCKET CONSISTENCY CHECK")
print("="*60)
for bucket in df['AQI_Bucket'].dropna().unique():
    sub = df[df['AQI_Bucket'] == bucket]['AQI']
    print(f"Bucket: {bucket:<15} Min AQI: {sub.min():>6.1f}, Max AQI: {sub.max():>6.1f}, Count: {len(sub):>5}")

print("\n" + "="*60)
print("DUPLICATE CITY/DATE CHECK")
print("="*60)
dup_count = df.duplicated(subset=['City', 'Date']).sum()
print(f"Duplicate (City, Date) rows in city_day.csv: {dup_count}")

print("\n" + "="*60)
print("TEMPORAL MISSINGNESS PER YEAR")
print("="*60)
df['Year'] = df['Date'].dt.year
for y, ydf in df.groupby('Year'):
    print(f"Year {y}: N={len(ydf):,}")
    for f in features_6 + ['AQI']:
        m_pct = (ydf[f].isnull().sum() / len(ydf)) * 100
        print(f"   {f:<6}: {m_pct:>5.1f}% missing", end=" | ")
    print()

