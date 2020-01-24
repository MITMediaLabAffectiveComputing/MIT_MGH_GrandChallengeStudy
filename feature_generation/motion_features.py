"""
    Author: Asma G
"""

import pandas as pd
from my_constants import MyConstants
from datetime import datetime

def update_date(series):
    tmp_date = datetime.strptime(series['date'], '%m/%d/%Y')
    series['date'] = tmp_date.strftime(MyConstants.DATE_FORMAT)
    series['average_motion_24hrs'] = (series['average_motion_24hrs_left']+series['average_motion_24hrs_right'])/2.0
    series['fraction_time_in_motion_24hrs'] = (series['fraction_time_in_motion_24hrs_left']+series['fraction_time_in_motion_24hrs_right'])/2.0
    series['median_motion_24hrs'] = (series['median_motion_24hrs_left']+series['median_motion_24hrs_right'])/2.0
    series['recording_time_fraction_24hrs'] = (series['recording_time_fraction_24hrs_left']+series['recording_time_fraction_24hrs_right'])/2.0
    series['std_motion_24hrs'] = (series['std_motion_24hrs_left']+series['std_motion_24hrs_right'])/2.0
    return series

def calculate_daily_motion():
    df = pd.read_csv(MyConstants.feature_dir+'intermediate_files/daily_motion.csv')
    # df = df.rename(columns={'Date': 'date'})
    df = df.apply(update_date, axis=1)
    return df

df_daily = calculate_daily_motion()
df_daily = df_daily.sort_values(['ID', 'date']).reset_index(drop=True)

df_daily.to_csv(MyConstants.feature_dir+'daily_motion.csv', index=False)