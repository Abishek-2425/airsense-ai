import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
city_day['Date'] = pd.to_datetime(city_day['Date'])

# Filter AQI > 500
aqi_gt_500 = city_day[city_day['AQI'] > 500].copy()
print(f"Total observations with AQI > 500: {len(aqi_gt_500)}")

# Breakdown by City
city_counts = aqi_gt_500['City'].value_counts()
print("\nAQI > 500 counts by City:")
print(city_counts)

# Summary statistics of AQI > 500
print("\nSummary statistics of AQI > 500:")
print(aqi_gt_500['AQI'].describe())

# Group by City: min, median, max, count
city_stats = aqi_gt_500.groupby('City')['AQI'].agg(['count', 'min', 'mean', 'median', 'max'])
print("\nCity-level AQI > 500 stats:")
print(city_stats)

# Date ranges for AQI > 500
for city in city_counts.index:
    subset = aqi_gt_500[aqi_gt_500['City'] == city]
    print(f"\n{city}: {len(subset)} rows, Dates: {subset['Date'].min().strftime('%Y-%m-%d')} to {subset['Date'].max().strftime('%Y-%m-%d')}")
    # Inspect top 3 rows
    top3 = subset.sort_values('AQI', ascending=False).head(3)
    for _, row in top3.iterrows():
        print(f"   Date: {row['Date'].strftime('%Y-%m-%d')}, AQI: {row['AQI']}, PM2.5: {row['PM2.5']}, PM10: {row['PM10']}, NO2: {row['NO2']}, CO: {row['CO']}, SO2: {row['SO2']}, O3: {row['O3']}")

# Let's inspect how CPCB sub-index is calculated
# CPCB breakpoints:
# Sub-index for pollutant p with concentration Cp in [B_low, B_high] mapping to [I_low, I_high]:
# Ip = I_low + ((I_high - I_low) / (B_high - B_low)) * (Cp - B_low)
# The highest breakpoint category in CPCB NAQI is 'Severe':
# PM2.5: 251-400+ -> AQI 401-500. Breakpoints: (250, 400) -> (400, 500). Slope = 100/150 = 0.6667.
# If linear extrapolation beyond 400 ug/m3 is used:
# AQI = 400 + (100/150) * (Cp - 250) = 500 + 0.6667 * (Cp - 400)
# Let's check CO:
# CO breakpoints (mg/m3):
# Good: 0-1 (0-50)
# Satisfactory: 1.1-2 (51-100)
# Moderate: 2.1-10 (101-200)
# Poor: 10.1-17 (201-300)
# Very Poor: 17.1-34 (301-400)
# Severe: 34.1-something? Wait! In CPCB NAQI (2014 report):
# Severe for CO: 34-something -> 401-500. Wait, what are the exact breakpoints for CO?
# Let's check what CPCB formula was used in Kaggle dataset!

