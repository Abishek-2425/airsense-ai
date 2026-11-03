import pandas as pd
df = pd.read_csv('archive/city_day.csv')
ahm = df[df['City'] == 'Ahmedabad'].dropna(subset=['AQI', 'CO'])
print("Correlation between CO and AQI in Ahmedabad:", ahm['CO'].corr(ahm['AQI']))
print("Correlation between PM2.5 and AQI in Ahmedabad:", ahm['PM2.5'].corr(ahm['AQI']))

# Compare with Delhi
delhi = df[df['City'] == 'Delhi'].dropna(subset=['AQI', 'CO'])
print("Correlation between CO and AQI in Delhi:", delhi['CO'].corr(delhi['AQI']))
print("Correlation between PM2.5 and AQI in Delhi:", delhi['PM2.5'].corr(delhi['AQI']))

# Let's check non-Ahmedabad AQI > 500
non_ahm_high_aqi = df[(df['City'] != 'Ahmedabad') & (df['AQI'] > 500)]
print(f"\nNon-Ahmedabad rows with AQI > 500: {len(non_ahm_high_aqi)} rows")
print(non_ahm_high_aqi[['City', 'Date', 'AQI', 'AQI_Bucket', 'PM2.5', 'PM10', 'CO']].head(15))

