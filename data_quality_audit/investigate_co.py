import pandas as pd
import numpy as np

df = pd.read_csv('archive/city_day.csv')
print("--- CO DISTRIBUTION BY CITY ---")
for city, grp in df.groupby('City'):
    c = grp['CO'].dropna()
    gt17 = (c > 17).sum()
    gt10 = (c > 10).sum()
    gt5 = (c > 5).sum()
    max_c = c.max() if len(c) > 0 else None
    med_c = c.median() if len(c) > 0 else None
    p95 = c.quantile(0.95) if len(c) > 0 else None
    if gt17 > 0 or (max_c and max_c > 5):
        print(f"{city:<20}: N={len(c):>4}, Med={med_c:>5.2f}, 95th={p95:>5.2f}, Max={max_c:>6.2f}, >5: {gt5:>3}, >10: {gt10:>3}, >17: {gt17:>3}")

print("\n--- AHMEDABAD CO VS ALL OTHER CITIES ---")
ahmedabad_co = df[df['City'] == 'Ahmedabad']['CO'].dropna()
other_co = df[df['City'] != 'Ahmedabad']['CO'].dropna()

print(f"Ahmedabad: count={len(ahmedabad_co)}, mean={ahmedabad_co.mean():.2f}, median={ahmedabad_co.median():.2f}, std={ahmedabad_co.std():.2f}, max={ahmedabad_co.max():.2f}")
print(f"  Ahmedabad > 10 count: {(ahmedabad_co > 10).sum()} ({(ahmedabad_co > 10).mean()*100:.1f}%)")
print(f"  Ahmedabad > 17 count: {(ahmedabad_co > 17).sum()} ({(ahmedabad_co > 17).mean()*100:.1f}%)")

print(f"\nOther 25 Cities: count={len(other_co)}, mean={other_co.mean():.2f}, median={other_co.median():.2f}, std={other_co.std():.2f}, max={other_co.max():.2f}")
print(f"  Other Cities > 10 count: {(other_co > 10).sum()} ({(other_co > 10).mean()*100:.2f}%)")
print(f"  Other Cities > 17 count: {(other_co > 17).sum()} ({(other_co > 17).mean()*100:.2f}%)")

print("\n--- OTHER CITIES WITH CO > 10 ---")
print(df[(df['City'] != 'Ahmedabad') & (df['CO'] > 10)][['City', 'Date', 'CO', 'PM2.5', 'AQI']])

