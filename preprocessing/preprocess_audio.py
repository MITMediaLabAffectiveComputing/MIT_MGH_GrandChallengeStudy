"""
    Author: Asma G
"""

import pandas as pd
import numpy as np
from datetime import datetime
from my_constants import MyConstants


#####################################################################################
#####################################       READ DATA           #####################
#####################################################################################

def update_date_format(series):
    series['date'] = datetime.strptime(series['date'], '%m/%d/%y').strftime('%Y-%m-%d')
    return series


def calc_total_affect(series):
    if np.isnan(series['average_pa1']) and np.isnan(series['average_pa2']):
        series['average_pa'] = np.nan
        series['average_na'] = np.nan
    else:
        series['average_pa'] = np.nanmean([series['average_pa1'], series['average_pa2']])
        series['average_na'] = np.nanmean([series['average_na1'], series['average_na2']])
    return series


def add_group(series):
    # add group information
    if series['ID'] in MyConstants.HC:
        series['group'] = 'HC'
    elif series['ID'] in MyConstants.MDD:
        series['group'] = 'MDD'
    else:
        series['group'] = 'DROPOUT'
    return series


def create_panas_dataset(id):
    print(id)
    df = pd.read_csv(MyConstants.survey_dir + id + '.csv')
    df['ID'] = id

    def calc_total_affect_score(series):
        series['average_pa1'] = (series['interested'] + series['excited'] + series['strong'] + series['enthusiastic'] + series['proud'])/5.0
        series['average_na1'] = (series['distressed'] + series['upset'] + series['guilty'] + series['scared'] + series['hostile'])/5.0
        series['average_pa2'] = (series['alert'] + series['inspired'] + series['determined'] + series['attentive'] + series['active'])/5.0
        series['average_na2'] = (series['irritable'] + series['ashamed'] + series['nervous'] + series['jittery'] + series['afraid'])/5.0
        series['form_duration'] = (datetime.strptime(series['Form_finish_time'], '%H:%M:%S')-datetime.strptime(series['Form_start_time'], '%H:%M:%S')).seconds
        return series

    df = df.apply(calc_total_affect_score, axis=1)

    df = df[['ID', 'Trigger_date', 'Form', 'form_duration', 'Missing', 'audio',
             'average_pa1', 'average_na1',  'average_pa2', 'average_na2']]

    return df


if __name__ == '__main__':
    all_df = pd.DataFrame()

    for user in MyConstants.HC+MyConstants.MDD:
        user_df = create_panas_dataset(user)
        all_df = pd.concat([all_df, user_df])
    all_df = all_df.reset_index(drop=True)

    #####################################################################################
    ##########       COMBINE DIFFERENT FORMS TO CREATE DAILY DATASET           ##########
    #####################################################################################

    audio_df = all_df[all_df['Form'] == 'Audio recording']
    audio_df = audio_df[['ID', 'Trigger_date', 'form_duration', 'Missing', 'audio']]
    audio_df = audio_df.rename(columns={'form_duration': 'audio_form_duration', 'Missing': 'audio_missing'})
    audio_df.drop_duplicates(['ID', 'Trigger_date'], inplace=True)

    feeling1_df = all_df[all_df['Form'] == 'Feeling1']
    feeling1_df = feeling1_df[['ID', 'Trigger_date', 'average_pa1', 'average_na1']]
    feeling1_df.drop_duplicates(['ID', 'Trigger_date'], inplace=True)

    feeling2_df = all_df[all_df['Form'] == 'Feeling2']
    feeling2_df = feeling2_df[['ID', 'Trigger_date', 'average_pa2', 'average_na2']]
    feeling2_df.drop_duplicates(['ID', 'Trigger_date'], inplace=True)

    clinical_df = pd.read_csv(MyConstants.data_dir + 'clinical/redcap_preprocessed.csv')
    clinical_df = clinical_df.dropna(subset=['date'])
    clinical_df = clinical_df.apply(update_date_format, axis=1)
    clinical_df = clinical_df[['ID', 'date', 'Name'] + MyConstants.ALL_LABEL_FACTORS]

    #combine all
    daily_df = feeling1_df.merge(feeling2_df, on=['ID', 'Trigger_date'], how='outer')
    daily_df = daily_df.apply(calc_total_affect, axis=1)
    daily_df.drop(['average_pa1', 'average_na1', 'average_pa2', 'average_na2'], inplace=True, axis=1)
    daily_df = daily_df.merge(audio_df, on=['ID', 'Trigger_date'], how='outer')
    daily_df = daily_df.rename(columns={'Trigger_date': 'date'})
    daily_df = daily_df.merge(clinical_df, on=['date', 'ID'], how='outer').reset_index(drop=True)
    daily_df = daily_df.apply(add_group, axis=1)
    daily_df = daily_df.sort_values(['group', 'ID', 'date'])
    daily_df = daily_df.dropna(subset=['date'])

    daily_df.to_csv(MyConstants.data_dir+'audio_metadata.csv', index=False)
