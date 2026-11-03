import pandas as pd

# Check columns and sample rows of station_hour.csv
sample_sh = pd.read_csv('archive/station_hour.csv', nrows=10)
print("station_hour columns:", sample_sh.columns.tolist())
print(sample_sh[['StationId', 'Datetime', 'PM2.5', 'AQI', 'AQI_Bucket']])

