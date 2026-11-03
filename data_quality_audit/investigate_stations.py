import pandas as pd
stations_df = pd.read_csv('archive/stations.csv')
print("--- STATIONS IN AHMEDABAD ---")
ahm_stations = stations_df[stations_df['City'] == 'Ahmedabad']
print(ahm_stations)

print("\n--- AHMEDABAD STATION READINGS IN station_day.csv ---")
# Check station_day for those stations
station_day_df = pd.read_csv('archive/station_day.csv', nrows=100)
print("station_day columns:", list(station_day_df.columns))

# Filter station_day for Ahmedabad station IDs
ahm_ids = ahm_stations['StationId'].tolist()
station_day_ahm = []
for chunk in pd.read_csv('archive/station_day.csv', chunksize=10000):
    sub = chunk[chunk['StationId'].isin(ahm_ids)]
    if len(sub) > 0:
        station_day_ahm.append(sub)

if station_day_ahm:
    s_ahm = pd.concat(station_day_ahm, ignore_index=True)
    print(f"Total station_day rows for Ahmedabad: {len(s_ahm)}")
    for sid, sgrp in s_ahm.groupby('StationId'):
        co_s = sgrp['CO'].dropna()
        print(f"Station {sid}: N={len(co_s)}, Mean CO={co_s.mean():.2f}, Med CO={co_s.median():.2f}, Max CO={co_s.max():.2f}")
else:
    print("No station_day rows found for Ahmedabad IDs!")

