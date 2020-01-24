"""
    Author: Asma G
"""

import pandas as pd
from datetime import datetime
import datetime as dt
import os
import xml.etree.ElementTree as ET
from my_constants import *

#####################################################################################
################        Clean, downsample, and impute location       #################
#####################################################################################


DIR = combined_log_dir


def clean_location():

    for dname in os.listdir(DIR):
        if dname.startswith('.') or '-' in dname or '_' in dname:
            continue
        print (dname)

        fname = DIR + dname + '/Location_edited.csv'
        if (os.stat(fname).st_size == 0):
            print('empty file')
            continue

        df = pd.read_csv(fname, header=None)
        df.columns = ['timestamp', 'lat', 'long', 'alt', 'acc', 'datetime']
        df.sort_values(by=['timestamp'], inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        df = df[df['acc'] < ACC_THRESHOLD]
        df = df.dropna(subset=['lat', 'long', 'datetime'])
        df.index = df['datetime']
        df = df.reset_index(drop=True)

        df_index = pd.Series(df.index, index=df['datetime'])
        resampled_index = df_index.resample('5T', fill_method='ffill', how=np.median)
        resampled_index = resampled_index.astype(int)
        resampled_df = df.loc[resampled_index.values]
        resampled_df['datetime'] = resampled_index.index
        resampled_df = resampled_df.reset_index(drop=True)
        # resampled_df = resampled_df.apply(date_index2str, axis=1)
        resampled_df['timestamp'] = resampled_df.index
        # resampled_df = resampled_df[['timestamp', 'lat', 'long', 'alt', 'acc', 'date']]
        resampled_df.to_csv(fname[0:-4] + '_resampled' + '.csv', index=False, header=None)


clean_location()
