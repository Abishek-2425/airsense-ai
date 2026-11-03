import pandas as pd
import numpy as np

station_day = pd.read_csv('archive/station_day.csv')
city_day = pd.read_csv('archive/city_day.csv')
stations = pd.read_csv('archive/stations.csv')

# Merge station_day with stations to get City
st_merged = station_day.merge(stations[['StationId', 'City']], on='StationId', how='left')

# Group station_day by City and Date, taking the mean of AQI
st_agg = st_merged.groupby(['City', 'Date'])['AQI'].mean().reset_index()
st_agg.rename(columns={'AQI': 'Station_Mean_AQI'}, inplace=True)

# Merge with city_day
cmp = city_day.merge(st_agg, on=['City', 'Date'], how='inner')
cmp = cmp.dropna(subset=['AQI', 'Station_Mean_AQI'])

cmp['Diff'] = (cmp['AQI'] - cmp['Station_Mean_AQI']).abs()
print("Comparison between city_day AQI and mean of station_day AQI:")
print("Number of overlapping city-date observations:", len(cmp))
print("Mean absolute difference:", cmp['Diff'].mean())
print("Max absolute difference:", cmp['Diff'].max())
print("Exact matches (within 1.0):", (cmp['Diff'] <= 1.0).sum(), "out of", len(cmp), f"({(cmp['Diff'] <= 1.0).mean()*100:.2f}%)")

