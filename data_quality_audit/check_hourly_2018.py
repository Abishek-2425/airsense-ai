import pandas as pd
for chunk in pd.read_csv('archive/city_hour.csv', chunksize=50000):
    sub = chunk[(chunk['City'] == 'Ahmedabad') & (chunk['Datetime'].str.startswith('2018-02-19'))]
    if len(sub) > 0:
        print(sub[['Datetime', 'CO', 'PM2.5', 'AQI']].head(10).to_string())
        print(sub[['Datetime', 'CO', 'PM2.5', 'AQI']].tail(5).to_string())
        break

