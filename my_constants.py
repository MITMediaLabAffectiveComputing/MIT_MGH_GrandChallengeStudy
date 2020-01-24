"""
    Author: Asma G
"""

import numpy as np
import pandas as pd
import os


class MyConstants:
    _SEED = None
    _TRAIN_TEST_OPTION = None  # 1,2,3 for prediction.

    IMPUTATION_AND_PREDICTION_LABEL = 'HAMD'  # SELECT FROM 'HAMD', 'HAMA', 'PSS', 'anxiety_factor', 'insomnia_factor',	'atypical_factor',	'melancholic_factor'
    IMPUTATION_INDIVIDUALIZATION = True  # individualized vs. not
    IMPUTATION_LABEL = IMPUTATION_AND_PREDICTION_LABEL  # HAMD vs. HAMA
    if (IMPUTATION_INDIVIDUALIZATION):
        IMPUTATION_LABEL_COMBINATION = 'individualized_' + IMPUTATION_LABEL
    else:
        IMPUTATION_LABEL_COMBINATION = 'original_' + IMPUTATION_LABEL

    PREDICTION_INDIVIDUALIZATION = True  # individualized vs. not
    PREDICTION_LABEL = IMPUTATION_AND_PREDICTION_LABEL  # individualized
    if (PREDICTION_INDIVIDUALIZATION):
        PREDICTION_LABEL_COMBINATION = 'individualized_' + PREDICTION_LABEL
    else:
        PREDICTION_LABEL_COMBINATION = 'original_' + PREDICTION_LABEL

    def create_needed_directories(inp):
        last_ind = 0
        while inp[last_ind:].find('/') != -1:
            directory = inp[0:last_ind + inp[last_ind:].find('/')]
            last_ind += inp[last_ind:].find('/') + 1
            if not os.path.exists(directory):
                os.makedirs(directory)

    HAMD_FILENAME = IMPUTATION_LABEL + '_imputed_survey_individualized.csv'  # 'HAMD_imputed_survey.csv' #'HAMD_imputed_survey_individualized.csv' #HAMD_imputed_survey
    ALL_LABEL_FACTORS = ['HAMD', 'HAMA', 'PSS', 'anxiety_factor', 'insomnia_factor', 'atypical_factor',
                         'melancholic_factor']

    LABEL_FACTORS = [
        'HAMD']  # , 'HAMA', 'PSS', 'anxiety_factor', 'insomnia_factor',	'atypical_factor',	'melancholic_factor']

    # INDIVIDUALIZATION=[True, False]
    # LABEL = ['HAMA', 'HAMD']

    def _update_directories(self):
        if self._SEED and self._TRAIN_TEST_OPTION:
            self.OPTIONS_SUBSTR = self.HAMD_FILENAME[:-4] + '/' + str(self._TRAIN_TEST_OPTION) + '/' + str(self._SEED)

            self.MODEL_FILE = 'factors/' + self.PREDICTION_LABEL + '_prediction/' + self.OPTIONS_SUBSTR + '/models/' + self.PREDICTION_LABEL_COMBINATION + '_model.txt'
            self.create_needed_directories(self.MODEL_FILE)

            self.IMPUTATION_MODEL_FILE = 'factors/' + self.IMPUTATION_LABEL + '_imputation/' + self.OPTIONS_SUBSTR + '/' + self.IMPUTATION_LABEL_COMBINATION + '_model.txt'
            self.create_needed_directories(self.IMPUTATION_MODEL_FILE)

            self.results_dir = 'factors/' + self.PREDICTION_LABEL + '_prediction/' + self.OPTIONS_SUBSTR + '/results/'
            self.create_needed_directories(self.results_dir)

            self.prediction_fig_dir = 'factors/' + self.PREDICTION_LABEL + '_prediction/' + self.OPTIONS_SUBSTR + '/figs/'
            self.create_needed_directories(self.prediction_fig_dir)
            self.create_needed_directories(self.prediction_fig_dir + 'test/')

            self.imputation_fig_dir = 'factors/' + self.IMPUTATION_LABEL + '_imputation/' + self.OPTIONS_SUBSTR + '/figs/'
            self.create_needed_directories(self.imputation_fig_dir)

    def set_HAMD_filename(self, hamd_filename):
        self.HAMD_FILENAME = hamd_filename
        # self._update_directories(self)

    def set_seed(self, SEED):
        self._SEED = SEED
        # self._update_directories(self)

    def get_seed(self):
        return self._SEED

    def set_train_test_option(self, TRAIN_TEST_OPTION):
        self._TRAIN_TEST_OPTION = TRAIN_TEST_OPTION
        # self._update_directories(self)

    def get_train_test_option(self):
        return self._TRAIN_TEST_OPTION

    _MODALITY = None  # ['ALL', 'call', 'sms', 'display', 'appUsage', 'location', 'motion_', 'sleep', 'HRV', 'eda_motionless', 'weather']
    MODALITIES = [['ALL', 'call', 'sms', 'display', 'appUsage', 'location', 'motion_', 'sleep', 'HRV', 'eda_motionless',
                   'weather']
        , ['PHONE_BASED', 'call', 'sms', 'display', 'appUsage', 'location', 'weather']
        , ['PHONE_ACC', 'call', 'sms', 'display', 'appUsage', 'location', 'weather', 'motion_', 'sleep']
        , ['ALL_BUT_EDA', 'call', 'sms', 'display', 'appUsage', 'location', 'motion_', 'sleep', 'HRV', 'weather']
        , ['ALL_BUT_RIGHT_EDA', 'left', 'call', 'sms', 'display', 'appUsage', 'location', 'motion_', 'sleep', 'HRV',
           'weather']
        , ['ALL_BUT_LEFT_EDA', 'right', 'call', 'sms', 'display', 'appUsage', 'location', 'motion_', 'sleep', 'HRV',
           'weather']
        , ['ALL_BUT_PHONE', 'location', 'weather', 'motion_', 'sleep', 'HRV', 'eda_motionless', 'weather']
        , ['ALL_BUT_LOCATION', 'call', 'sms', 'display', 'appUsage', 'motion_', 'sleep', 'HRV', 'eda_motionless']
        , ['ALL_BUT_MOTION', 'call', 'sms', 'display', 'appUsage', 'location', 'motion_', 'sleep', 'HRV',
           'eda_motionless', 'weather']
        , ['ALL_BUT_SLEEP', 'right', 'call', 'sms', 'display', 'appUsage', 'location', 'sleep', 'HRV', 'weather']
        , ['ALL_BUT_HRV', 'call', 'sms', 'display', 'appUsage', 'location', 'motion_', 'sleep', 'eda_motionless',
           'weather']
        , ['PHONE', 'appUsage', 'call', 'sms', 'display']
        , ['LOCATION', 'location', 'weather']
        , ['MOTION', 'motion_']
        , ['SLEEP', 'sleep']
        , ['EDA', 'eda_motionless']
        , ['HRV', 'HRV']
        , ['DEBUG']]

    def set_modality(self, MODALITY):
        for modality in self.MODALITIES:
            if modality[0] == MODALITY:
                self._MODALITY = modality
                return

    def get_modality(self):
        return self._MODALITY

    USER_SEED = 1

    MAX_PCA_ALL = 25
    MAX_PCA_SUB = 20
    MAX_PCA_SUB_HIST = 25


    # 1: train on imputed, test on actual
    # 2: train on some users, test on hold-out users
    # 3: train on early weeks, test on last two weeks

    HC = [] # replace with the list of HC users
    MDD = [] # replace with the list of MDD users
    outliers = [] # replace with the list of participants who dropped out.
    NO_OUTGOING_SMS = [] # replace with the list of participants with no outgoing messages
    NEEDS_CLEANING = [] # replace with the list of participants that switched phone/movisens username in the middle of the study

    data_dir = 'data/'
    survey_dir = data_dir + 'raw_survey/'
    feature_dir = data_dir + 'features/'
    raw_log_dir = data_dir + 'raw_logs/'
    combined_log_dir = data_dir + 'combined_logs/'

    ACC_THRESHOLD = 1000 # radius accuracy in meters for location data
    CALL_TYPES = ['Incoming',  'IncomingMissed',  'IncomingDismissed',
                  'Outgoing', 'OutgoingNotReached']
    SMS_TYPES = ['Incoming', 'Outgoing']
    APP_TYPES = ['game', 'email', 'web', 'calendar', 'communication', 'facebook',
                 'maps', 'youtube', 'photo', 'shopping', 'clock']


    K_FOLD_N = 10
    TEST_RATIO = 0.1
    STATIONARY_SPEED = 0.3
    HOME_DISTANCE_THRESHOLD = 500


    # for dimensionality reduction
    EXPLAINED_VARIANCE_THRESHOLD = 0.85

    REGULARIZATION_ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0]
    def get_app_type(app):
        if app in ['air.com.sgn.bookoflife.gp']:
            return 'game'
        if app in ['com.android.email', 'com.yahoo.mobile.client.android.mail']:
            return 'email'
        if app in ['com.android.chrome', 'com.sec.android.app.sbrowser']:
            return 'web'
        if app in ['com.android.calendar']:
            return 'calendar'
        if app in ['com.android.contacts', 'com.android.incallui', 'com.android.mms', 'com.android.phone', 'com.whatsapp']:
            return 'communication'
        if app in ['com.facebook.katana', 'com.facebook.orca']:
            return 'facebook'
        if app in ['com.google.android.apps.maps']:
            return 'maps'
        if app in ['com.google.android.youtube']:
            return 'youtube'
        if app in ['com.sec.android.app.camera', 'com.sec.android.gallery3d', 'com.sec.android.mimage.photoretouching']:
            return 'photo'
        if app in ['com.walmart.android', 'com.target.ui', 'com.macys.android']:
            return 'shopping'
        if app in ['com.sec.android.app.clockpackage']:
            return 'clock'


    #game, email, web, calendar, communication, facebook, maps, video streaming, photo, shopping, clock

    intervals = [[0, 6, '0_to_6'], [6, 12, '6_to_12'], [12, 18, '12_to_18'],
                [18, 24, '18_to_24'], [9, 18, 'day_hours'], [0, 24, 'daily']]
    for i in range(24):
        intervals +=[[i, i+1, str(i)+'_to_'+str(i+1)]]

    DATE_FORMAT = '%Y-%m-%d'
    def date_index2str(self, series):
        series['date'] = series['datetime'].strftime(self.DATE_FORMAT)
        return series
    #sensor_data_dir='sleep_sensor_data/'


    def convert_one_hot_str(df, col):
        cols = np.unique(df[col])
        old_col = np.array(df[col])
        for c in cols:
            new_col = []
            for i in old_col:
                if i == c:
                    new_col.append(1)
                else:
                    new_col.append(0)

            df[col+'_'+c] = new_col

        return df


    CALL_SUB_FEATRURES = ['call_daily_IncomingDismissed_count_call',
                        'call_daily_IncomingMissed_count_call',
                        'call_daily_Incoming_count_call',
                        'call_daily_Incoming_mean_call_duration',
                        'call_daily_Incoming_std_call_duration',
                        'call_daily_Outgoing_count_call',
                        'call_daily_Outgoing_mean_call_duration',
                        'call_daily_Outgoing_std_call_duration',
                        'call_daily_incoming_outgoing_call_duration',
                        'call_daily_incoming_outgoing_call_count']
    DISPLAY_SUB_FEATURES = ['display_daily_sum_on_duration',
                            'display_daily_std_on_duration',
                            'display_daily_mean_on_duration',
                            'display_daily_median_on_duration',
                            'display_daily_count_on']
    LOCATION_SUB_FEATURES = ['location_day_hours_weighted_avg_std',
                             'location_daily_weighted_avg_std',
                             'weighted_stationary_avg_std',
                             'weighted_home_stay',
                             'transition_time',
                             'total_distance']
    SLEEP_SUB_FEATURES = ['sleep_24hrs_fraction_recording',
                            'sleep_24hrs_sleep_(s)',
                            'sleep_night_sleep_(s)',
                            'sleep_night_fraction_recording',
                            'sleep_night_sleep_onset_timeelapsed_since_noon_(s)',
                            'sleep_night_max_uninterrupted_sleep_(s)',
                            'sleep_night_nbwakeups',
                            'sleep_ day_wakeup_onset_timeelapsed_since_midnight_(s)',
                            'sleep_sleep_reg_index']
    SMS_SUB_FEATURES = ['sms_daily_Incoming_count_sms']
    MOTION_SUB_FEATURES = ['motion_average_motion_24hrs',
                           'motion_fraction_time_in_motion_24hrs',
                           'motion_median_motion_24hrs',
                           'motion_recording_time_fraction_24hrs',
                           'motion_std_motion_24hrs']

    EDA_SUB_FEATURES = ['eda_motionless_eda_motionless_24hrs_mean_difference_r_l',
                        'eda_motionless_eda_motionless_24hrs_mean_left',
                        'eda_motionless_eda_motionless_24hrs_mean_right',
                        'eda_motionless_eda_motionless_24hrs_scl_cvx_mean_diff_r_l',
                        'eda_motionless_eda_motionless_24hrs_scl_cvx_mean_left',
                        'eda_motionless_eda_motionless_24hrs_scl_cvx_mean_right',
                        'eda_motionless_eda_motionless_24hrs_scr_cvx_mean_diff_r_l',
                        'eda_motionless_eda_motionless_24hrs_scr_cvx_mean_left',
                        'eda_motionless_eda_motionless_24hrs_scr_cvx_mean_right',
                        'eda_motionless_eda_motionless_24hrs_scrs_diff_r_l',
                        'eda_motionless_eda_motionless_24hrs_scrs_left',
                        'eda_motionless_eda_motionless_24hrs_scrs_mean_ampl_diff_r_l',
                        'eda_motionless_eda_motionless_24hrs_scrs_mean_ampl_left',
                        'eda_motionless_eda_motionless_24hrs_scrs_mean_ampl_right',
                        'eda_motionless_eda_motionless_24hrs_scrs_right',
                        'eda_motionless_eda_motionless_night_mean_difference_r_l',
                        'eda_motionless_eda_motionless_night_mean_left',
                        'eda_motionless_eda_motionless_night_mean_right',
                        'eda_motionless_eda_motionless_night_scl_cvx_mean_diff_r_l',
                        'eda_motionless_eda_motionless_night_scl_cvx_mean_left',
                        'eda_motionless_eda_motionless_night_scl_cvx_mean_right',
                        'eda_motionless_eda_motionless_night_scr_cvx_mean_diff_r_l',
                        'eda_motionless_eda_motionless_night_scr_cvx_mean_left',
                        'eda_motionless_eda_motionless_night_scr_cvx_mean_right',
                        'eda_motionless_eda_motionless_night_scrs_diff_r_l',
                        'eda_motionless_eda_motionless_night_scrs_left',
                        'eda_motionless_eda_motionless_night_scrs_mean_ampl_diff_r_l',
                        'eda_motionless_eda_motionless_night_scrs_mean_ampl_left',
                        'eda_motionless_eda_motionless_night_scrs_mean_ampl_right',
                        'eda_motionless_eda_motionless_night_scrs_right']
    WEATHER_SUB_FEATURES = [] #TODO: add sub weather features if needed


    ONE_HOT_USERS = ['ID_'+user for user in MDD] 

    SUB_FEATURES = CALL_SUB_FEATRURES+DISPLAY_SUB_FEATURES+LOCATION_SUB_FEATURES+\
                   SLEEP_SUB_FEATURES+SMS_SUB_FEATURES+MOTION_SUB_FEATURES+EDA_SUB_FEATURES+\
                   ONE_HOT_USERS+WEATHER_SUB_FEATURES




    #### REMOVE HC s
    def add_group(series):
        if (series['ID']in HC):
            series['group']='HC'
        else:
            series['group']='MDD'
        return series

    indices = pd.read_csv(data_dir + HAMD_FILENAME)

    indices = indices.apply(add_group, axis=1)
    indices = indices[indices['group']=='MDD'].reset_index(drop=True)

    IND_TRAIN = list(indices[indices['imputed']=='y'].index)
    IND_TEST = list(indices[indices['imputed']=='n'].index)

    def split_data_ind(self, inds, test_N):

        # OPTION 1: train on imputed, test on actual
        if (self._TRAIN_TEST_OPTION == 1):
            indices = pd.read_csv(self.data_dir + self.HAMD_FILENAME)
            no_imputation = len(indices[indices['imputed'] == 'y']) == 0
            if no_imputation:
                inds = list(indices.index)
                np.random.seed(self.USER_SEED)
                np.random.shuffle(inds)
                testsize = int(len(inds) / 5.0)
                ind_train = inds[testsize:]
                ind_test = inds[0:testsize]
                np.random.seed(self._SEED)
            else:
                ind_train = list(indices[indices['imputed'] == 'y'].index)
                ind_test = list(indices[indices['imputed'] == 'n'].index)

        # OPTION 2: train on some users (imputed + actual) , test on actual data from other users (actual)
        elif (self._TRAIN_TEST_OPTION == 2):

            indices = pd.read_csv(self.data_dir + self.HAMD_FILENAME)
            USERS = np.unique(indices['ID'])
            np.random.seed(self.USER_SEED)
            np.random.shuffle(USERS)
            np.random.seed(self._SEED)
            TEST_USER_IND = int(self.TEST_RATIO*len(USERS))+2
            TEST_USERS = USERS[0:TEST_USER_IND]
            TRAIN_USERS = USERS[TEST_USER_IND:]

            ind_train = []
            for user in TRAIN_USERS:
                ind_train.extend(list(indices[indices['ID'] == user].index))

            ind_test = []
            for user in TEST_USERS:
                user_df = indices[indices['ID'] == user]
                ind_test.extend(list(user_df[user_df['imputed']=='n'].index))

        # OPTION 3: train on data upto week 5 , test on actual data from last 2 actual data points
        elif (self._TRAIN_TEST_OPTION == 3):

            indices = pd.read_csv(self.data_dir + self.HAMD_FILENAME)
            USERS = np.unique(indices['ID'])

            ind_train = []
            ind_test = []
            no_imputation = len(indices[indices['imputed']=='y']) == 0
            for user in USERS:
                user_df = indices[indices['ID'] == user]
                actual_user_df = user_df[user_df['imputed'] == 'n']
                if (len(actual_user_df)>2):
                    ind_test.extend(list(actual_user_df.index)[-2:])
                    tmp_train = list(user_df.index)
                    for ind in tmp_train:
                        if no_imputation: #if there's no imputation, just use all the previous data
                            ind_train.extend(list(actual_user_df.index)[:-2])
                        elif (ind<list(actual_user_df.index)[-2]-7): # else if there is imputation, throw away 7 days in between
                            ind_train.append(ind)

        return ind_train, ind_test



    def standardize_df(df):
        remove_col = []
        for i in range(len(df.columns.values)):
            if np.std(df[df.columns[i]]) == 0:
                remove_col.append(i)
        x_df = df.drop(df.columns[remove_col], axis=1)
        x = np.array(x_df)
        x = (x - np.mean(x, 0)) / np.std(x, 0)

        x_df_standardized = x_df
        for i in range(len(x_df.columns)):
            x_df_standardized[x_df.columns.values[i]] = x[:, i]
        return x_df_standardized




    GROUPING_VARIABLES = ['sleep_24hrs_sleep_(s)',
    'sleep_night_sleep_onset_timeelapsed_since_noon_(s)',
    'sleep_sleep_reg_index',
    'location_smart_stationary_total_std',
    'location_smart_home_stay',
    'location_smart_total_distance',
    'eda_motionless_eda_motionless_24hrs_mean_left_normalized',
    'eda_motionless_eda_motionless_24hrs_scrs_left_normalized',
    'eda_motionless_eda_motionless_24hrs_scrs_diff_r_l_normalized',
    'eda_motionless_eda_motionless_24hrs_mean_difference_r_l_normalized',
    'call_daily_IncomingDismissed_count_call',
    'call_daily_IncomingMissed_count_call',
    'call_daily_Incoming_count_call',
    'call_daily_Incoming_mean_call_duration',
    'call_daily_Outgoing_count_call',
    'call_daily_Outgoing_mean_call_duration',
    'display_0_to_6_count_on',
    'display_0_to_6_mean_on_duration',
    'sms_daily_Incoming_count_sms',
    'sms_daily_Outgoing_count_sms',
    'HRV_hr_mean_24hrs_left',
    'HRV_hr_std_24hrs_left',
    'HRV_mean_rr_AVNN_24hrs_left',
    'HRV_pNN50_24hrs_left',
    'HRV_rMSSD_24hrs_left']
