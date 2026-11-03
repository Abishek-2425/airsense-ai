import pandas as pd
df = pd.read_csv('archive/city_day.csv')
ahm = df[df['City'] == 'Ahmedabad'].copy()
ahm['Date'] = pd.to_datetime(ahm['Date'])
ahm['Year'] = ahm['Date'].dt.year

print("--- AHMEDABAD CO YEAR-BY-YEAR ---")
for y, ygrp in ahm.groupby('Year'):
    c = ygrp['CO'].dropna()
    print(f"Year {y}: N={len(c)}, Mean={c.mean():.2f}, Median={c.median():.2f}, Min={c.min():.2f}, Max={c.max():.2f}, >17 count={(c > 17).sum()}")

print("\n--- AHMEDABAD AQI YEAR-BY-YEAR ---")
for y, ygrp in ahm.groupby('Year'):
    aqi = ygrp['AQI'].dropna()
    print(f"Year {y}: N={aqi.count()}, Mean AQI={aqi.mean():.1f}, Median AQI={aqi.median():.1f}, Max AQI={aqi.max():.1f}, AQI > 500={(aqi > 500).sum()}")

