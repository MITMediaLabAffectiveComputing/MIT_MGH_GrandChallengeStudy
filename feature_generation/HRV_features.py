"""
    Author: Asma G
"""

import pandas as pd
from my_constants import MyConstants
from datetime import datetime

def update_date(series):
    tmp_date = datetime.strptime(series['date'], '%m/%d/%Y')
    series['date'] = tmp_date.strftime(MyConstants.DATE_FORMAT)
    return series
def calculate_daily_sleep():
    df = pd.read_csv(MyConstants.feature_dir+'intermediate_files/daily_HRV.csv')
    # df = df.rename(columns={'Unnamed: 0': 'date'})
    df = df.apply(update_date, axis = 1)
    return df

df_daily = calculate_daily_sleep()
df_daily = df_daily.sort_values(['ID', 'date']).reset_index(drop=True)

df_daily.to_csv(MyConstants.feature_dir+'daily_HRV.csv', index=False)