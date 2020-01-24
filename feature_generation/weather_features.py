"""
    Author: Asma G
"""

import pandas as pd
import numpy as np
import os
from my_constants import MyConstants

#######################        Daily weather features using resampled location       #######################
# dataframe columns:
# 'ID', 'datetime', 'lat_std', 'lat_mean', 'lat_median', 'long_std', 'long_mean', 'long_median', 'total_std'

DIR = MyConstants.feature_dir + 'intermediate_files/daily_weather/'


def calculate_daily_weather():
    df_all = pd.DataFrame()
    for fname in os.listdir(DIR):
        if fname.startswith('.'):
            continue
        user = fname[0:4]
        df = pd.read_csv(DIR + fname)
        df = df.rename(columns={'Unnamed: 0': 'datetime', 'id': 'ID'})
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', utc=True)
        df = df.apply(MyConstants.date_index2str, axis=1)
        df.sort_values(by=['date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df_all = df_all.append(df, ignore_index=True)
        print(user + ' done')

    df_all.drop(columns=['pressureError'], inplace=True)
    df_all.fillna(0, inplace=True)
    df_all.drop(columns=['datetime'], inplace=True)
    return df_all


df_daily = calculate_daily_weather()
df_daily = df_daily.sort_values(['ID', 'date']).reset_index(drop=True)


cols = df_daily.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
cols.insert(0, cols.pop(cols.index('date')))
df_daily = df_daily.reindex(columns=cols)

df_daily.to_csv(MyConstants.feature_dir+'daily_weather.csv', index=False)