import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
city_day['Date'] = pd.to_datetime(city_day['Date'])
train_full = city_day[city_day['Date'] <= pd.to_datetime('2019-12-04')].copy()

co_train = train_full['CO'].dropna()

print("CO Training Distribution (All Cities):")
print(co_train.describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.995, 0.999]))

q1 = co_train.quantile(0.25)
q3 = co_train.quantile(0.75)
iqr = q3 - q1
iqr_3 = q3 + 3 * iqr
iqr_5 = q3 + 5 * iqr
p99 = co_train.quantile(0.99)
p995 = co_train.quantile(0.995)

print(f"\nQ1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
print(f"Q3 + 3*IQR: {iqr_3:.2f}")
print(f"Q3 + 5*IQR: {iqr_5:.2f}")
print(f"99th Percentile: {p99:.2f}")
print(f"99.5th Percentile: {p995:.2f}")
print(f"Statutory CPCB Threshold: 17.00 mg/m3")

# Observations exceeding thresholds in training:
print(f"\nTraining observations > 17 mg/m3: {(co_train > 17.0).sum()}")
print(f"Training observations > Q3 + 3*IQR ({iqr_3:.2f}): {(co_train > iqr_3).sum()}")
print(f"Training observations > 99th percentile ({p99:.2f}): {(co_train > p99).sum()}")

# Non-Ahmedabad training CO:
co_non_ahm = train_full[train_full['City'] != 'Ahmedabad']['CO'].dropna()
print("\nCO Training Distribution (Excluding Ahmedabad):")
print(co_non_ahm.describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.995, 0.999]))
print(f"Non-Ahmedabad Max CO: {co_non_ahm.max():.2f}")
print(f"Non-Ahmedabad 99.9th percentile: {co_non_ahm.quantile(0.999):.2f}")

