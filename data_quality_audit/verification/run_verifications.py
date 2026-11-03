import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")

# Load datasets
city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
city_day['Date'] = pd.to_datetime(city_day['Date'])
city_hour = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_hour.csv'), low_memory=False)
city_hour['Datetime'] = pd.to_datetime(city_hour['Datetime'])
station_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'station_day.csv'))
station_day['Date'] = pd.to_datetime(station_day['Date'])
stations = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'stations.csv'))

print("\n" + "="*70)
print("PART 1: VERIFY AHMEDABAD STATIONS & CO CONTAMINATION")
print("="*70)

# Check all stations in Gujarat and Ahmedabad
ahm_stations = stations[stations['City'] == 'Ahmedabad']
print("Stations with City == 'Ahmedabad' in stations.csv:")
print(ahm_stations.to_string())

guj_stations = stations[stations['State'] == 'Gujarat']
print("\nStations in State == 'Gujarat':")
print(guj_stations.to_string())

# Check station_day for Ahmedabad
print("\nUnique StationIds in station_day.csv with City == Ahmedabad:")
ahm_station_days = station_day[station_day['StationId'].isin(ahm_stations['StationId'])]
print(f"Total station_day rows for Ahmedabad: {len(ahm_station_days)}")
print(f"StationId counts: {ahm_station_days['StationId'].value_counts().to_dict()}")

# Check if city_day and station_day match exactly for Ahmedabad
cd_ahm = city_day[city_day['City'] == 'Ahmedabad'].sort_values('Date').reset_index(drop=True)
sd_ahm = ahm_station_days[ahm_station_days['StationId'] == 'GJ001'].sort_values('Date').reset_index(drop=True)

print(f"city_day Ahmedabad rows: {len(cd_ahm)}")
print(f"station_day GJ001 rows:  {len(sd_ahm)}")

# Compare values
cols_to_compare = ['CO', 'PM2.5', 'NO2', 'SO2', 'O3', 'AQI']
match_diffs = {}
for col in cols_to_compare:
    diff = (cd_ahm[col] - sd_ahm[col]).abs().dropna()
    max_d = diff.max() if len(diff) > 0 else 0
    match_diffs[col] = max_d
print(f"Max absolute difference between city_day and station_day GJ001: {match_diffs}")

# Check hourly observations for GJ001 / Ahmedabad
# In city_hour, how many non-null hours per day for CO in Ahmedabad?
ahm_hour = city_hour[city_hour['City'] == 'Ahmedabad'].copy()
ahm_hour['Date'] = ahm_hour['Datetime'].dt.floor('D')

daily_hour_stats = ahm_hour.groupby('Date').agg(
    co_count=('CO', 'count'),
    co_mean=('CO', 'mean'),
    co_max=('CO', 'max'),
    pm25_count=('PM2.5', 'count'),
    pm25_mean=('PM2.5', 'mean')
).reset_index()

# Merge with city_day
merged_ahm = pd.merge(cd_ahm, daily_hour_stats, on='Date', how='left')

print("\n--- Daily Aggregation vs Hourly Observations in Ahmedabad ---")
# Check how many extreme CO days (>17 mg/m3) had <16 hours of CO data
ext_co_days = merged_ahm[merged_ahm['CO'] > 17]
print(f"Total days with CO > 17 mg/m3 in Ahmedabad: {len(ext_co_days)}")
print(f"Days with CO > 17 mg/m3 having < 16 hours data: {(ext_co_days['co_count'] < 16).sum()} ({(ext_co_days['co_count'] < 16).mean()*100:.1f}%)")
print(f"Days with CO > 17 mg/m3 having < 12 hours data: {(ext_co_days['co_count'] < 12).sum()}")
print(f"Days with CO > 17 mg/m3 having < 8 hours data:  {(ext_co_days['co_count'] < 8).sum()}")

# Check difference between city_day CO and mean of city_hour CO
merged_ahm['co_diff'] = (merged_ahm['CO'] - merged_ahm['co_mean']).abs()
print(f"Max difference between city_day CO and mean(city_hour CO): {merged_ahm['co_diff'].max():.6f}")

# Check multi-pollutant correlations in Ahmedabad vs other cities
print("\n--- Multi-Pollutant Correlation Matrix in Ahmedabad ---")
ahm_corr = cd_ahm[['CO', 'PM2.5', 'NO2', 'SO2', 'O3', 'AQI']].corr()
print(ahm_corr.round(3).to_string())

print("\n--- Multi-Pollutant Correlation Matrix across all other cities ---")
other_cities_corr = city_day[city_day['City'] != 'Ahmedabad'][['CO', 'PM2.5', 'NO2', 'SO2', 'O3', 'AQI']].corr()
print(other_cities_corr.round(3).to_string())

# Check whether other pollutants spike when CO spikes in Ahmedabad
print("\n--- Other Pollutants during Ahmedabad CO Spikes ---")
ahm_normal_co = cd_ahm[cd_ahm['CO'] <= 10]
ahm_ext_co = cd_ahm[cd_ahm['CO'] > 50]
print(f"Ahmedabad normal CO (<=10 mg/m3, N={len(ahm_normal_co)}):")
print(ahm_normal_co[['PM2.5', 'NO2', 'SO2', 'O3', 'AQI']].mean().round(2).to_dict())
print(f"Ahmedabad extreme CO (>50 mg/m3, N={len(ahm_ext_co)}):")
print(ahm_ext_co[['PM2.5', 'NO2', 'SO2', 'O3', 'AQI']].mean().round(2).to_dict())

print("\nDone Part 1.\n")

