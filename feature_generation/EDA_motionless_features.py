"""
    Author: Asma G
"""

import pandas as pd
from my_constants import MyConstants
from datetime import datetime

def update_date(series):
    tmp_date = datetime.strptime(series['date'], '%m/%d/%y')
    series['date'] = tmp_date.strftime(MyConstants.DATE_FORMAT)
    return series
def calculate_daily_eda_motionless():
    df = pd.read_csv(MyConstants.feature_dir+'intermediate_files/daily_eda_motionless.csv')
    # df = df.rename(columns={'Unnamed: 0': 'date'})
    df = df.apply(update_date, axis=1)
    return df

df_daily = calculate_daily_eda_motionless()
df_daily = df_daily.sort_values(['ID', 'date']).reset_index(drop=True)

df_daily.to_csv(MyConstants.feature_dir+'daily_eda_motionless.csv', index=False)