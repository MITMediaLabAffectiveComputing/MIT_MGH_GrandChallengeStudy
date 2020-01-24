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
def create_PANAS_dataset(group, ID):
    print (group, ID)
    df = pd.read_csv(MyConstants.survey_dir+ID+'.csv')
    df = df[df['Form'] != 'Settings']
    df = df[df['Form'] != 'Audio recording']
    df['ID'] = ID
    df['group'] = group
    def calc_total_affect_score(series):
        series['total_PA1'] = (series['interested'] + series['excited'] + series['strong'] + series['enthusiastic'] + series['proud'])/5.0
        series['total_NA1'] = (series['distressed'] + series['upset'] + series['guilty'] +series['scared'] + series['hostile'])/5.0
        series['total_PA2'] = (series['alert'] + series['inspired'] + series['determined'] + series['attentive'] + series['active'])/5.0
        series['total_NA2'] = (series['irritable'] + series['ashamed'] + series['nervous'] + series['jittery'] + series['afraid'])/5.0
        series['form_duration'] = (datetime.strptime(series['Form_finish_time'], '%H:%M:%S')-datetime.strptime(series['Form_start_time'], '%H:%M:%S')).seconds
        return series

    df = df.apply(calc_total_affect_score, axis=1)

    df.drop(['Participant','Trigger', 'Trigger_counter', 'Form_upload_date', 'Form_upload_time', 'audio',  'emotionsTime1-2', 'emotionsTime2-2',
                'active2', 'afraid2', 'attentive2', 'determined2', 'enthusiastic2', 'guilty2', 'hostile2', 'jittery2', 'proud2', 'scared2',
               'beerImage', 'ciderImage', 'coffeePic', 'energyPic', 'sodaPic', 'spiritsImage', 'teaPic', 'wineImage',
                'thankQuote', 'passcode_1', 'passcode_2', 'passcode_3', 'passcode_4',
               'passcode_text', 'passcode_wrong_text',  'settings_save_text'], inplace=True, axis=1)
    return df


all_df = pd.DataFrame()



for user in MyConstants.HC:
    user_df = create_PANAS_dataset('HC', user)
    all_df = pd.concat([all_df, user_df])

for user in MyConstants.MDD:
    user_df = create_PANAS_dataset('MDD', user)
    all_df = pd.concat([all_df, user_df])

all_df = all_df.reset_index(drop = True)
all_df.drop(['Form_start_date', 'Form_start_time', 'Form_finish_date', 'Form_finish_time',
             'weeklyMeditationTime', 'weeklyMeditation', 'morning_time', 'evening_time'], inplace=True, axis=1)






#####################################################################################
##########       COMBINE DIFFERENT FORMS TO CREATE DAILY DATASET           ##########
#####################################################################################
BEER_ALC = 4.6
WINE_ALC = 12.0
SPIRITS_ALC = 40.0
CIDER_ALC = 5.0
ALC_CONVERTOR = 60.0

COFFEE_CAFFEINE = 16 #doesn't count for espresso :| 70
TEA_CAFFEINE = 6
SODA_CAFFEINE = 4
ENERGY_CAFFEINE = 10
# CAFFEINE_CONVERTOR = 200 t0 300 mg caffeine is ok during day

def calc_total_affect(series):
    if (np.isnan(series['total_PA1']) and np.isnan(series['total_PA2'])):
        series['total_PA'] = np.nan
        series['total_NA'] = np.nan
    else:
        series['total_PA'] = np.nanmean([series['total_PA1'], series['total_PA2']])
        series['total_NA'] = np.nanmean([series['total_NA1'], series['total_NA2']])
    return series

def calc_total_beverage(series):
    series['total_alc'] = (BEER_ALC*series['beerAmount'] + WINE_ALC*series['wineAmount']+\
    SPIRITS_ALC*series['spiritsAmount'] + CIDER_ALC*series['ciderAmount'])/ALC_CONVERTOR

    series['total_caffeine'] = (COFFEE_CAFFEINE*series['coffeeAmount'] + TEA_CAFFEINE*series['teaAmount']+\
    SODA_CAFFEINE*series['sodaAmount'] + ENERGY_CAFFEINE*series['energyAmount'])
    return series

morning_df = all_df[all_df['Form']=='Morning']
morning_df = morning_df[['ID', 'group', 'Trigger_date', 'form_duration', 'Trigger_time', 'Missing', 'sleepInteractionTime',\
                         'sleepLog', 'morningTime1', 'wakingEarly', 'generalSleepQuality', 'morningTime2', 'morningSad',\
                         'morningGuilty', 'morningView', 'concentrationMorning', 'morningSlowedDown', 'morningCold']]
morning_df = morning_df.rename(columns={'form_duration': 'morning_form_duration',
                                       'Missing': 'morning_missing',
                                       'Trigger_time': 'morning_trigger_time'})

beverages_df = all_df[all_df['Form']=='Beverages']
beverages_df = beverages_df[['ID', 'group', 'Trigger_date', 'form_duration', 'Trigger_time', 'Missing', 'beverage_1',\
                             'beverage_2', 'beverage_3', 'beverage_4', 'beverage_5', 'beverage_6', 'beverage_7',\
                             'beverage_8', 'beerAmount', 'beerT', 'wineAmount', 'wineT','spiritsAmount', 'spiritsT',\
                             'ciderAmount', 'ciderT', 'coffeeAmount', 'coffeeT','teaAmount', 'teaT', 'sodaAmount', 'sodaT',\
                             'energyAmount', 'energyT']]
beverages_df[['beerAmount', 'wineAmount', 'spiritsAmount', 'ciderAmount',
              'coffeeAmount', 'teaAmount', 'sodaAmount', 'energyAmount']] =\
    beverages_df[['beerAmount', 'wineAmount', 'spiritsAmount', 'ciderAmount',
                  'coffeeAmount', 'teaAmount', 'sodaAmount', 'energyAmount']]\
        .fillna(0)
beverages_df = beverages_df.apply(calc_total_beverage, axis=1)
beverages_df = beverages_df.rename(columns={'form_duration': 'beverages_form_duration',
                                           'Missing': 'beverages_missing',
                                           'Trigger_time': 'beverages_trigger_time'})

medication_df = all_df[all_df['Form']=='Medication']
medication_df = medication_df[['ID', 'group', 'Trigger_date', 'form_duration', 'Trigger_time', 'Missing', 'medication_1',\
                               'medication_2', 'medication_3', 'medication_4', 'sleepMed','anxietyMed', 'painMed', 'otherMed']]
medication_df = medication_df.rename(columns={'form_duration': 'medication_form_duration',
                                             'Missing': 'medication_missing',
                                             'Trigger_time': 'medication_trigger_time'})


midday_df = all_df[all_df['Form']=='midday'] #if the same time as Feeling1 or Feeling2
midday_df = midday_df[['ID', 'group', 'Trigger_date', 'form_duration', 'Trigger_time', 'Missing', 'middaySocial']]

feeling1_df = all_df[all_df['Form']=='Feeling1']
feeling1_df = feeling1_df[['ID', 'group', 'Trigger_date', 'form_duration', 'Trigger_time', 'Missing', 'emotionsTime1', \
                           'interested', 'distressed', 'excited', 'upset', 'guilty', 'strong', 'scared', 'hostile',\
                           'enthusiastic', 'proud', 'total_PA1', 'total_NA1' ]]
feeling1_df = feeling1_df.rename(columns={'form_duration': 'feeling1_form_duration',
                                         'Missing': 'feeling1_missing'})
feeling1_df = feeling1_df.merge(midday_df, on=['group', 'ID', 'Trigger_date', 'Trigger_time'], how='inner')
feeling1_df = feeling1_df.rename(columns={'form_duration': 'midday1_form_duration',
                                         'Missing': 'midday1_missing',
                                         'Trigger_time': 'feeling1_trigger_time',
                                         'middaySocial': 'middaySocial1'})
feeling1_df.drop_duplicates(['ID', 'group', 'Trigger_date'], inplace=True)

feeling2_df = all_df[all_df['Form']=='Feeling2']
feeling2_df = feeling2_df[['ID', 'group', 'Trigger_date', 'form_duration', 'Trigger_time', 'Missing', 'emotionsTime2',\
                           'irritable', 'alert', 'ashamed', 'inspired', 'nervous', 'determined', 'attentive', 'jittery',\
                           'active', 'afraid' ,'total_PA2', 'total_NA2' ]]
feeling2_df = feeling2_df.rename(columns={'form_duration': 'feeling2_form_duration',
                                         'Missing': 'feeling2_missing'})
feeling2_df = feeling2_df.merge(midday_df, on=['group', 'ID', 'Trigger_date', 'Trigger_time'], how='inner')
feeling2_df = feeling2_df.rename(columns={'form_duration': 'midday2_form_duration',
                                         'Missing': 'midday2_missing',
                                         'Trigger_time': 'feeling2_trigger_time',
                                         'middaySocial': 'middaySocial2'})
feeling2_df.drop_duplicates(['ID', 'group', 'Trigger_date'], inplace=True)


evening_df = all_df[all_df['Form']=='Evening']
evening_df = evening_df[['ID', 'group', 'Trigger_date', 'form_duration', 'Trigger_time', 'Missing', 'eveningTime1',\
                         'fruits', 'supplements', 'appetiteChange', 'appetiteChangeAmount', 'nap', 'napDuration',\
                         'eveningTime2', 'eveningSad', 'eveningGuitly', 'eveningView', 'joyfulEvent', 'stressfulEvent',\
                         'eveningTime3', 'meditation']]
evening_df = evening_df.rename(columns={'form_duration': 'evening_form_duration',
                                       'Missing': 'evening_missing',
                                       'Trigger_time': 'evening_trigger_time'})



#combine all
daily_df = morning_df.merge(beverages_df, on=['group', 'ID', 'Trigger_date'], how='outer')
daily_df = daily_df.merge(medication_df, on=['group', 'ID', 'Trigger_date'], how='outer')
daily_df = daily_df.merge(feeling1_df, on=['group', 'ID', 'Trigger_date'], how='outer')
daily_df = daily_df.merge(feeling2_df, on=['group', 'ID', 'Trigger_date'], how='outer')
daily_df = daily_df.merge(evening_df, on=['group', 'ID', 'Trigger_date'], how='outer')
daily_df = daily_df.rename(columns={'Trigger_date': 'date'})
daily_df = daily_df.sort_values(['group', 'ID', 'date'])
daily_df = daily_df.dropna(subset = ['date'])
daily_df = daily_df.apply(calc_total_affect, axis=1)
daily_df = daily_df.reset_index(drop=True)


daily_df.to_csv(MyConstants.feature_dir+'daily_survey.csv', index=False)