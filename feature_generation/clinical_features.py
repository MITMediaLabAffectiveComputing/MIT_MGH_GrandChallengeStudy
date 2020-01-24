"""
    Author: Asma G
"""

import pandas as pd
CLINICAL_DATA_DIR = 'data/clinical/'
STANDARD_SURVEYS = ['hamd', 'hama', 'pss', 'saftee', 'ers', 'rrs', 'qids']

def fill_0(inp, length):
    curr_length = len(str(inp))
    return '0'*(length-curr_length)+str(inp)

def preprocess_df(series):
    series['ID'] = 'M'+fill_0(series['record_id'], 3)
    event_name = series['redcap_event_name']
    if 'screen' in event_name:
        series['Name'] = 'Screen'
        series['week'] = -1
    else:
        week = event_name[event_name.find('_')+1:event_name.find('_')+2]
        series['Name'] = 'Week '+week
        series['week'] = week
    series['date'] = series['date_of_visit']
    series['HAMD'] = series['hamd17_total']
    series['PSS'] = series['pss_total_score']
    series['HAMA'] = series['hama_total_score']
    anxiety_indices = [10, 11, 12, 13, 15, 17] # >=7 has anxiety
    atypical_indices = [13, 22, 23, 24, 25, 26]
    melancholic_indices =[1, 2, 6, 7, 8, 9, 12, 16, '18b']
    insomnia_indices = [4, 5, 6]
    series['anxiety_factor'] = 0
    series['atypical_factor'] = 0
    series['melancholic_factor'] = 0
    series['insomnia_factor'] = 0
    for ind in anxiety_indices:
        series['anxiety_factor'] += series['hamd_'+str(ind)]
    for ind in atypical_indices:
        series['atypical_factor'] += series['hamd_' + str(ind)]
    for ind in melancholic_indices:
        series['melancholic_factor'] += series['hamd_'+str(ind)]
    for ind in insomnia_indices:
        series['insomnia_factor'] += series['hamd_'+str(ind)]
    return series

def create_clinical_features():
    df = pd.read_csv(CLINICAL_DATA_DIR+'redcap_raw.csv')
    cols = df.columns.values
    cols_subset = ['record_id', 'redcap_event_name', 'date_of_visit', 'hamd17_total']
    for c in cols:
        if 'complete' in c:
            continue
        if 'comments' in c:
            continue
        if c =='hamd_mdd':
            continue
        if 'partners' in c:
            continue
        for survey_name in STANDARD_SURVEYS:
            if survey_name+'_' in c:
                cols_subset.append(c)
    print (cols_subset)
    df = df[cols_subset]
    df = df.apply(preprocess_df, axis=1)
    df = df.sort_values(['ID', 'week']).reset_index(drop=True)
    df.to_csv(CLINICAL_DATA_DIR+'redcap_preprocessed.csv', index=False)

create_clinical_features()