import pandas as pd
import os

print("--- AHMEDABAD HOURLY DATA FOR 2017-10-25 ---")
# Check city_hour.csv
target_found = []
for chunk in pd.read_csv('archive/city_hour.csv', chunksize=50000):
    sub = chunk[(chunk['City'] == 'Ahmedabad') & (chunk['Datetime'].str.startswith('2017-10-25'))]
    if len(sub) > 0:
        target_found.append(sub)

if target_found:
    df_h = pd.concat(target_found, ignore_index=True)
    print(df_h[['Datetime', 'CO', 'PM2.5', 'PM10', 'NO2', 'AQI']].to_string())
else:
    print("No matching rows in city_hour.csv!")

