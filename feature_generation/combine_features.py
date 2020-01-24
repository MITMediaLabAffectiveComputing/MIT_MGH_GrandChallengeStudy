"""
    Author: Asma G
"""

import pandas as pd
import os
from my_constants import MyConstants


def combine_features():
    all_features_df = pd.DataFrame()
    for fname in os.listdir(MyConstants.feature_dir):
        if fname.startswith('.') or fname == 'daily_all.csv' or fname == 'daily_survey.csv' or fname == 'daily_eda.csv': #because we are using motionless
            continue
        if fname == 'daily_location_smart.csv' or fname == 'daily_location.csv':
            continue
        if 'incomplete' in fname:
            continue
        if 'intermediate_files' in fname:
            continue
        print (fname)
        feature = fname[fname.find('_')+1:-4]
        feature_df = pd.read_csv(MyConstants.feature_dir+fname)
        feature_df.columns = [feature+'_'+col for col in feature_df.columns.values]
        feature_df = feature_df.rename(columns={feature+'_date': 'date', feature+'_ID': 'ID'})
        if len(all_features_df)==0:
            all_features_df = all_features_df.append(feature_df, ignore_index=True)
        else:
            all_features_df = all_features_df.merge(feature_df, on=['date', 'ID'], how='outer')
    all_features_df = all_features_df.sort_values(['ID', 'date']).reset_index(drop=True)
    all_features_df.to_csv(MyConstants.feature_dir+'daily_all.csv', index=False)

combine_features()