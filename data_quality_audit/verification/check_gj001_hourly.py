import pandas as pd

# Read a chunk or query station_hour for GJ001 on 2018-02-19
chunks = []
for chunk in pd.read_csv('archive/station_hour.csv', chunksize=100000):
    subset = chunk[(chunk['StationId'] == 'GJ001') & (chunk['Datetime'].str.startswith('2018-02-19'))]
    if len(subset) > 0:
        chunks.append(subset)
if chunks:
    gj_hours = pd.concat(chunks)
    print("GJ001 on 2018-02-19 hours:")
    print(gj_hours[['Datetime', 'PM2.5', 'NO2', 'CO', 'AQI', 'AQI_Bucket']])
    print("Mean hourly AQI:", gj_hours['AQI'].mean())

