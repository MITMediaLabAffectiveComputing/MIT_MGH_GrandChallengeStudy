"""
    Author: Asma G
"""

from my_constants import MyConstants
import numpy as np
import pandas as pd
from datetime import datetime


#######################################################################################
###########################       COMBINE SURVEY AND CLINICAL DATA  ###################
#######################################################################################

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return ((d1 - d2).days)


def update_date_format(series):
    series['date'] = datetime.strptime(series['date'], '%m/%d/%y').strftime('%Y-%m-%d')
    return series

def add_group(series):
    # add group information
    if series['ID'] in MyConstants.HC:
        series['group'] = 'HC'
    elif series['ID'] in MyConstants.MDD:
        series['group'] = 'MDD'
    else:
        series['group'] = 'INVALID_USER'
    return series

def calc_day(series):
    # calculate day
    user = HAMD[HAMD['ID']==series['ID']]
    offset = user[user['Name']=='Week 0']
    offset = offset['date'].iloc[0]
    series['day'] = days_between(series['date'], offset)
    series['weekday'] = datetime.strptime(series['date'], "%Y-%m-%d").weekday()
    return series



def calc_avg_mood(series):
    # calculate mood
    user = HAMD[HAMD['ID']==series['ID']]
    user['day']-=series['day']
    prev_week = user[user['day']<=0]
    prev_week = prev_week[prev_week['day']>-7]
    prev_week.dropna(subset=['total_PA', 'total_NA'], inplace=True)
    prev_week['weighted_PA'] = prev_week['total_PA']*2.0**prev_week['day']
    prev_week['weighted_NA'] = prev_week['total_NA']*2.0**prev_week['day']
    series['avg_weekly_PA'] = np.nanmean(prev_week['total_PA'])
    series['avg_weekly_NA'] = np.nanmean(prev_week['total_NA'])
    if len(prev_week)>0:
        series['weighted_avg_weekly_PA'] = np.sum(prev_week['weighted_PA'])/np.sum(2.0**prev_week['day'])
        series['weighted_avg_weekly_NA'] = np.sum(prev_week['weighted_NA'])/np.sum(2.0**prev_week['day'])
    else:
        series['weighted_avg_weekly_PA'] = np.nan
        series['weighted_avg_weekly_NA'] = np.nan

    if not np.isnan(series['total_PA']):
        series['total_NA/PA'] = series['total_NA']/series['total_PA']
    else:
        series['total_NA/PA'] = np.nan

    if not np.isnan(series['avg_weekly_PA']):
        series['avg_weekly_NA/PA'] = series['avg_weekly_NA']/series['avg_weekly_PA']
    else:
        series['avg_weekly_NA/PA'] = np.nan

    if not np.isnan(series['weighted_avg_weekly_PA']):
        series['weighted_avg_weekly_NA/PA'] = series['weighted_avg_weekly_NA']/series['weighted_avg_weekly_PA']
    else:
        series['weighted_avg_weekly_NA/PA'] = np.nan

    series['std_weekly_PA'] = np.nanstd(prev_week['total_PA'])
    series['std_weekly_NA'] = np.nanstd(prev_week['total_NA'])

    series['avg_overall_PA'] = np.nanmean(user['total_PA'])
    series['avg_overall_NA'] = np.nanmean(user['total_NA'])
    if not np.isnan(series['avg_overall_PA']):
        series['avg_overall_NA/PA'] = series['avg_overall_NA']/series['avg_overall_PA']
    else:
        series['avg_overall_NA/PA'] = np.nan

    series['std_overall_PA'] = np.nanstd(user['total_PA'])
    series['std_overall_NA'] = np.nanstd(user['total_NA'])

    return series


HAMD = pd.read_csv(MyConstants.data_dir+'clinical/redcap_preprocessed.csv')
HAMD = HAMD.dropna(subset=['date'])
HAMD = HAMD.apply(update_date_format, axis=1)
HAMD = HAMD[['ID','date','Name']+MyConstants.LABEL_FACTORS]

daily_df = pd.read_csv(MyConstants.feature_dir+'daily_survey.csv')
HAMD = HAMD.merge(daily_df, on =['date', 'ID'], how='outer').reset_index(drop=True)

HAMD = HAMD.apply(add_group, axis=1)
HAMD = HAMD[HAMD['group'] != 'INVALID_USER'].reset_index(drop=True)
for outlier in MyConstants.outliers:
    HAMD = HAMD[HAMD['ID'] != outlier]
HAMD = HAMD.apply(calc_day, axis=1)
HAMD = HAMD.apply(calc_avg_mood, axis=1)
HAMD = HAMD.sort_values(['ID', 'day']).reset_index(drop=True)



HAMD.to_csv(MyConstants.data_dir+'daily_survey_HAMD.csv', index=False)


########################################################################################
###########################       CALCULATE DEVIATION FROM BASELINE  ###################
########################################################################################


#HAMD = pd.read_csv(data_dir+'daily_survey_HAMD.csv')
print (HAMD.columns)

def calc_var_from_screen(df):
    HAMD_cols = ['ID', 'date', 'group', 'Name']+['individualized_'+col for col in MyConstants.LABEL_FACTORS]+['original_' + col for col in MyConstants.LABEL_FACTORS]+['baseline_' + col for col in MyConstants.LABEL_FACTORS]
    survey_cols = [col for col in list(df.columns) if col not in HAMD_cols and col not in MyConstants.LABEL_FACTORS]
    updated_df = pd.DataFrame(columns=HAMD_cols+survey_cols)
    users = np.unique(df['ID'])
    for user in users:
        user_df = df[df['ID']==user]
        for label in MyConstants.LABEL_FACTORS:
            baseline_label = user_df.sort_values(by=['date']).reset_index(drop=True)
            baseline_label = baseline_label[label][0]

            user_df['original_' + label] = user_df[label]
            user_df['individualized_'+label] = user_df[label] - baseline_label
            user_df['baseline_'+label] = baseline_label

        # baseline_HAMD = user_df.sort_values(by=['date']).reset_index(drop=True)
        # baseline_HAMD = baseline_HAMD['HAMD'][0]
        #
        # baseline_HAMA = user_df.sort_values(by=['date']).reset_index(drop=True)
        # baseline_HAMA = baseline_HAMA['HAMA'][0]
        #
        # user_df['individualized_HAMD'] = user_df['HAMD'] - baseline_HAMD
        #
        # user_df['individualized_HAMA'] = user_df['HAMA'] - baseline_HAMA
        #
        # user_df['baseline_HAMD'] = baseline_HAMD
        #
        # user_df['baseline_HAMA'] = baseline_HAMA

        user_df = user_df.reset_index(drop=True)
        updated_df = pd.concat([updated_df, user_df])
    updated_df = updated_df[HAMD_cols + survey_cols]
    return updated_df

updated_df = calc_var_from_screen(HAMD)
# updated_df = updated_df.rename(columns={"HAMD": "original_HAMD", "HAMA": "original_HAMA"})
updated_df = updated_df.reset_index(drop=True)
updated_df.to_csv(MyConstants.data_dir+'daily_survey_HAMD_individualized.csv', index=False)