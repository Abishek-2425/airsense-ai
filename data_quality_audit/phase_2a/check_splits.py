import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

city_day = pd.read_csv(os.path.join(PROJECT_ROOT, 'archive', 'city_day.csv'))
city_day['Date'] = pd.to_datetime(city_day['Date'])
city_day = city_day.sort_values(['City', 'Date']).reset_index(drop=True)

# Shift target
city_day['Next_Date'] = city_day.groupby('City')['Date'].shift(-1)
city_day['Target_AQI'] = city_day.groupby('City')['AQI'].shift(-1)

mask_1day = (city_day['Next_Date'] - city_day['Date']) == pd.Timedelta(days=1)
valid_paired = city_day[mask_1day].dropna(subset=['Target_AQI']).copy()
valid_paired = valid_paired.sort_values('Date').reset_index(drop=True)

print(f"Total paired supervised samples: {len(valid_paired)}")
print(f"Date range: {valid_paired['Date'].min().strftime('%Y-%m-%d')} to {valid_paired['Date'].max().strftime('%Y-%m-%d')}")

# Candidate / Production split:
# Train: Date <= 2019-12-04
# Embargo: Date == 2019-12-05
# Test: Date >= 2019-12-06
train_full = valid_paired[valid_paired['Date'] <= pd.to_datetime('2019-12-04')].copy().reset_index(drop=True)
embargo = valid_paired[valid_paired['Date'] == pd.to_datetime('2019-12-05')].copy().reset_index(drop=True)
test_full = valid_paired[valid_paired['Date'] >= pd.to_datetime('2019-12-06')].copy().reset_index(drop=True)

print(f"\nTrain full (<= 2019-12-04): {len(train_full)} records ({train_full['Date'].min().strftime('%Y-%m-%d')} to {train_full['Date'].max().strftime('%Y-%m-%d')})")
print(f"Embargo (2019-12-05): {len(embargo)} records")
print(f"Test full (>= 2019-12-06): {len(test_full)} records ({test_full['Date'].min().strftime('%Y-%m-%d')} to {test_full['Date'].max().strftime('%Y-%m-%d')})")

# Let's also look at an internal train/validation split within train_full for tuning:
# Train split: Date <= 2018-12-31
# Embargo: Date == 2019-01-01
# Val split: 2019-01-02 to 2019-12-04
train_sub = train_full[train_full['Date'] <= pd.to_datetime('2018-12-31')].copy().reset_index(drop=True)
embargo_val = train_full[train_full['Date'] == pd.to_datetime('2019-01-01')].copy().reset_index(drop=True)
val_sub = train_full[train_full['Date'] >= pd.to_datetime('2019-01-02')].copy().reset_index(drop=True)

print(f"\nInternal Train (<= 2018-12-31): {len(train_sub)} records")
print(f"Internal Embargo (2019-01-01): {len(embargo_val)} records")
print(f"Internal Validation (2019-01-02 to 2019-12-04): {len(val_sub)} records")

