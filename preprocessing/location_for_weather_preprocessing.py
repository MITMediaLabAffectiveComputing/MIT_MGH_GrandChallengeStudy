"""
    Author: Asma G
"""

import pandas as pd
from datetime import datetime
import datetime as dt
import os
from my_constants import *
from scipy import stats

#####################################################################################
################        Clean, downsample, and impute location       #################
#####################################################################################


DIR = combined_log_dir

def create_loc_pair(series):
    series['loc'] = (series['lat'], series['long'])
    return series

def clean_location():
    df_all = pd.DataFrame()
    for dname in os.listdir(DIR):
        if dname.startswith('.') or '-' in dname or '_' in dname:
            continue
        print (dname)

        fname = DIR + dname + '/Location_edited_resampled.csv'
        if (os.stat(fname).st_size == 0):
            print('empty file')
            continue

        df = pd.read_csv(fname, header=None)
        df.columns = ['timestamp', 'lat', 'long', 'alt', 'acc', 'datetime']
        # df.sort_values(by=['timestamp'], inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        # df = df[df['acc'] < ACC_THRESHOLD]
        # df = df.dropna(subset=['lat', 'long', 'datetime'])
        df = df.apply(date_index2str, axis=1)
        df = df.apply(create_loc_pair, axis=1)

        # grouped = df[['lat', 'long', 'loc', 'date']].groupby(['date']).agg(stats.mode)
        # print (grouped)
        dates = np.unique(df['date'])
        user_df = pd.DataFrame()
        for dt in dates:
            print('day ' + dt)
            day_df = df[df['date'] == dt]
            mode = day_df.mode().iloc[0]
            most_frequent = pd.Series(data={'lat': mode['loc'][0], 'long': mode['loc'][1], 'date': dt})
            user_df = user_df.append(most_frequent, ignore_index=True)
        user_df['ID'] = dname
        df_all = df_all.append(user_df, ignore_index=True)
    return df_all





df = clean_location()
df = df.sort_values(by=['ID', 'date'])

cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
cols.insert(0, cols.pop(cols.index('date')))
df = df.reindex(columns=cols)
df.to_csv(feature_dir+'location_for_weather.csv', index=False)
