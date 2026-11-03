import os
import pandas as pd
import numpy as np

# CPCB Breakpoints:
# PM2.5: [0, 30, 60, 90, 120, 250, 400] -> [0, 50, 100, 200, 300, 400, 500]
# For Cp > 250: I_low=401, I_high=500, B_low=251, B_high=400 (or 250 to 400)
# PM10: [0, 50, 100, 250, 350, 430, 500] -> [0, 50, 100, 200, 300, 400, 500]
# CO: [0, 1.0, 2.0, 10.0, 17.0, 34.0, ???] 
# Let's check what the standard Kaggle / Indian AQI calculation code does.

def calc_pm25_subindex(x):
    if pd.isna(x): return np.nan
    if x <= 30: return x * 50 / 30
    elif x <= 60: return 50 + (x - 30) * 50 / 30
    elif x <= 90: return 100 + (x - 60) * 100 / 30
    elif x <= 120: return 200 + (x - 90) * 100 / 30
    elif x <= 250: return 250 + (x - 120) * 100 / 130 # wait, let's verify standard breakpoints
    else: return 400 + (x - 250) * 100 / 150 # linear extrapolation above 250/400

# Let's inspect city_day.csv and station_day.csv to see if AQI is already provided in the original data or computed
station_day = pd.read_csv('archive/station_day.csv')
print("Station_day columns:", station_day.columns.tolist())
print("Null AQI in station_day:", station_day['AQI'].isnull().sum(), "out of", len(station_day))

# Check Delhi 2016-11-07
delhi_row = station_day[(station_day['StationId'].str.startswith('DL')) & (station_day['Date'] == '2016-11-07')]
print("\nDelhi stations on 2016-11-07:")
print(delhi_row[['StationId', 'PM2.5', 'PM10', 'AQI']])

# Check Ahmedabad station on 2018-02-19
ahm_row = station_day[(station_day['StationId'] == 'GJ001') & (station_day['Date'] == '2018-02-19')]
print("\nGJ001 on 2018-02-19:")
print(ahm_row[['StationId', 'PM2.5', 'NO2', 'CO', 'AQI']])

