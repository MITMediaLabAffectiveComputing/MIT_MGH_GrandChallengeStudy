"""
    Author: Asma G
"""

import pandas as pd
import numpy as np
import os
from my_constants import MyConstants

#######################        Daily dataset for all users resampled location       #######################
# dataframe columns:
# 'ID', 'datetime', 'lat_std', 'lat_mean', 'lat_median', 'long_std', 'long_mean', 'long_median', 'total_std'

DIR = MyConstants.combined_log_dir



def calculate_daily_location(intervals):

    df_all = pd.DataFrame()
    for dname in os.listdir(DIR):
        if dname.startswith('.') or '-' in dname or '_' in dname:
            continue


        fname = DIR+dname+'/Location_edited_resampled.csv'
        df = pd.read_csv(fname, header=None)
        df.columns = ['timestamp', 'lat', 'long', 'alt', 'acc', 'datetime']
        df.sort_values(by=['timestamp'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        df = df.apply(MyConstants.date_index2str, axis=1)
        df = df.dropna(subset=['lat', 'long', 'date'])
        df.reset_index(drop=True, inplace=True)

        user_df = pd.DataFrame()
        for intrvl in intervals:
            MIN_HOUR = intrvl[0]
            MAX_HOUR = intrvl[1]
            intrvl_name = intrvl[2]

            mask = df.datetime.apply(lambda x: x.hour>=MIN_HOUR and x.hour<MAX_HOUR)
            masked_df = df.loc[mask]
            masked_df.sort_values(by=['timestamp'], inplace=True)
            masked_df.reset_index(drop=False, inplace=True)

            grouped = masked_df[['lat', 'long', 'date']].groupby(['date'])

            df_agg = grouped.agg([np.nanstd, np.nanmean, np.nanmedian])
            df_agg.columns = [' '.join(col).strip() for col in df_agg.columns.values]
            df_agg.columns = [intrvl_name+'_lat_std', intrvl_name+'_lat_mean', intrvl_name+'_lat_median', intrvl_name+'_long_std', intrvl_name+'_long_mean', intrvl_name+'_long_median']

            if len(df_agg) == 0:
                continue;

            df_agg[intrvl_name + '_avg_std'] = (df_agg[intrvl_name + '_lat_std'] + df_agg[
                intrvl_name + '_long_std']) / 2.0
            df_agg['ID'] = dname
            df_agg.reset_index(level=0, inplace=True)
            if len(user_df) == 0:
                user_df = user_df.append(df_agg, ignore_index=True)
            else:
                user_df = user_df.merge(df_agg, on=['date', 'ID'], how='outer')
        print (dname + ' done')


        df_all = df_all.append(user_df, ignore_index=True)


    return df_all


df_daily = calculate_daily_location(MyConstants.intervals)
df_daily = df_daily.sort_values(['ID', 'date']).reset_index(drop=True)


cols = df_daily.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
cols.insert(0, cols.pop(cols.index('date')))
df_daily = df_daily.reindex(columns=cols)

df_daily.to_csv(MyConstants.feature_dir+'daily_location_resampled.csv', index=False)
