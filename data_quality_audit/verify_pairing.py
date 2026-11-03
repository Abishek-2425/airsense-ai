import pandas as pd

df = pd.read_csv('archive/city_day.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['City', 'Date']).reset_index(drop=True)

df['Next_Date'] = df.groupby('City')['Date'].shift(-1)
df['Next_City'] = df.groupby('City')['City'].shift(-1)
df['Target_AQI'] = df.groupby('City')['AQI'].shift(-1)

# Mask for consecutive calendar days
mask_1day = (df['Next_Date'] - df['Date']) == pd.Timedelta(days=1)
paired_df = df[mask_1day].dropna(subset=['Target_AQI']).copy()

print(f"Total paired rows: {len(paired_df)}")
# Assertions
assert (paired_df['City'] == paired_df['Next_City']).all(), "City mismatch in pair!"
assert ((paired_df['Next_Date'] - paired_df['Date']) == pd.Timedelta(days=1)).all(), "Date delta is not 1 day!"
assert paired_df['Target_AQI'].notnull().all(), "Null target in paired rows!"

print("All pairing assertions passed!")

# Check embargo in candidate model
train_cand = paired_df[paired_df['Date'] <= pd.to_datetime('2019-12-04')]
embargo = paired_df[paired_df['Date'] == pd.to_datetime('2019-12-05')]
test_cand = paired_df[paired_df['Date'] >= pd.to_datetime('2019-12-06')]

print(f"Candidate Train rows: {len(train_cand)}, Max Date: {train_cand['Date'].max()}, Max Next_Date: {train_cand['Next_Date'].max()}")
print(f"Embargo rows: {len(embargo)}, Date: {embargo['Date'].unique()}")
print(f"Candidate Test rows: {len(test_cand)}, Min Date: {test_cand['Date'].min()}, Min Next_Date: {test_cand['Next_Date'].min()}")

assert train_cand['Next_Date'].max() < test_cand['Date'].min(), "Train target overlaps test date!"
print("Embargo assertion passed: Max train target date (2019-12-05) < Min test date (2019-12-06)!")

