"""
    Author: Asma G
"""

import pandas as pd
from my_constants import MyConstants

def update_date(series):
    series['date'] = series['date'][0:10]
    return series
def calculate_daily_sleep():
    df = pd.read_csv(MyConstants.feature_dir+'intermediate_files/daily_sleep.csv')
    # df = df.rename(columns={'Unnamed: 0': 'date'})
    df = df.apply(update_date, axis = 1)
    df.drop(['24hrs_sleep', 'night_sleep', 'night_sleep_onset_timeelapsed_since_noon',
             'night_max_uninterrupted_sleep', 'day_wakeup_onset_timeelapsed_since_midnight'], inplace=True, axis=1)
    return df

df_daily = calculate_daily_sleep()
df_daily = df_daily.sort_values(['ID', 'date']).reset_index(drop=True)

df_daily.to_csv(MyConstants.feature_dir+'daily_sleep.csv', index=False)