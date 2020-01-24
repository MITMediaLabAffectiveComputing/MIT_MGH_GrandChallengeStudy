"""
    File name: e4_files_downloader.py
    Author: Szymon F
    Description: The script generates EDA features from the EDA measurements collected on both wrists with E4 sensors.
    The input are the hdf files generated with the e4files_downloader_gcs_v4 script
"""


import pandas as pd
from datetime import datetime,timedelta
import scipy.signal as scisig
import numpy as np
import biosppy.signals.eda as eda
import cvxEDA
from scipy.signal import butter, lfilter


user_id = 'M011'
data_folder_path = 'E:\Grand-Chalenge study\Data\Completed Participants\h5_streams'
eda_filepath = data_folder_path + '\\' + user_id + '\\' + user_id + '_eda.h5'
temp_filepath = data_folder_path + '\\' + user_id + '\\' + user_id + '_temp.h5'
acc_filepath = data_folder_path + '\\' + user_id + '\\' + user_id + '_acc.h5'

eda_features_filepath = 'P:\Grand Challenge Funding\Analysis\eda analysis\eda_features_33-34.csv'
eda_motionless_features_filepath = 'P:\Grand Challenge Funding\Analysis\eda analysis\eda_motionless_features.csv'

filter_out_based_on_temp_threshold = True
min_valid_temp_value = 30
fs_eda_e4 = 4

active_motion_threshold = 0.05
fs_acc_e4 = 32#Hz
subsampling_factor = 4

min_motionless_eda_segment_len = fs_eda_e4 * 30#seconds


def butter_bandpass(lcut, hcut, f_s, order=5):
    nyq = 0.5 * f_s
    low = lcut / nyq
    high = hcut / nyq
    b, a = butter(order, [low, high], btype='band', analog = 'false')
    return b, a

def apply_butter_filter(df):
    # lowcut = 0.1
    # highcut = 16
    lowcut = 0.6875
    highcut = 10


    b, a = butter_bandpass(lowcut, highcut, fs_acc_e4)
    df_out = lfilter(b, a, df)
    return df_out





data_folder_path = 'E:\Grand-Chalenge study\Data\Completed Participants\h5_streams'
acc_filepath = data_folder_path + '\\' + user_id + '\\' + user_id + '_acc.h5'

# split into 24hrs
# calculate eda features over 24hrs when no motion
# calculate eda featuers during night
# calculate eda features first 30mins after onset sleep

def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

def read_hdfs_and_calculate_features(acc_path, eda_path, temp_path, start_time, end_time):
    acc_stream_left_from_file = pd.read_hdf(acc_path, 'ACC_left',
                                            where='index>=start_time & index<end_time & columns = z')
    print ('Finished reading Left Acc')
    acc_stream_right_from_file = pd.read_hdf(acc_path, 'ACC_right',
                                             where='index>=start_time & index<end_time & columns = z')
    print ('Finished reading Right Acc')

    eda_stream_left_from_file = pd.read_hdf(eda_path, 'EDA_left',
                                            where='index>=start_time & index<end_time')
    print ('Finished reading Left EDA')
    eda_stream_right_from_file = pd.read_hdf(eda_path, 'EDA_right',
                                             where='index>=start_time & index<end_time')
    print ('Finished reading Right EDA')

    temp_stream_left_from_file = pd.read_hdf(temp_path, 'TEMP_left',
                                            where='index>=start_time & index<end_time')
    print ('Finished reading Left TEMP')
    temp_stream_right_from_file = pd.read_hdf(temp_path, 'TEMP_right',
                                             where='index>=start_time & index<end_time')
    print ('Finished reading Right TEMP')

    eda_stream_left_from_file_and_temp = eda_stream_left_from_file.join(temp_stream_left_from_file, how='left')
    eda_stream_right_from_file_and_temp = eda_stream_right_from_file.join(temp_stream_right_from_file, how='left')

    if filter_out_based_on_temp_threshold:
        eda_stream_left_from_file_and_temp = eda_stream_left_from_file_and_temp[eda_stream_left_from_file_and_temp.temp > min_valid_temp_value]
        eda_stream_right_from_file_and_temp = eda_stream_right_from_file[eda_stream_right_from_file_and_temp.temp > min_valid_temp_value]

    df_features = pd.DataFrame( )
    df_features.ix[start_time, 'ID'] = user_id

    max_24hrs_signal_length = 24 * 60 * 60 * fs_eda_e4
    max_morning_afternoon_evening_night_signal_length = 6 * 60 * 60 * fs_eda_e4 - 1

    # calculate eda features for left hand stream and merged (R vs L) streams
    if len(eda_stream_left_from_file_and_temp.index) < 21:
        print ('There is no recording from Left hand for the specified time interval!')
    else:
        # calculate 24hrs EDA features
        eda_24hrs_mean_left, eda_24hrs_scrs_left, eda_24hrs_scrs_mean_ampl_left, eda_24hrs_scl_cvx_mean_left, \
            eda_24hrs_scr_cvx_mean_left = calculate_one_hand_eda_features(eda_stream_left_from_file_and_temp)

        if len(eda_stream_right_from_file_and_temp.index) > 20:
            df_eda_24hrs_merged = eda_stream_left_from_file_and_temp.join(eda_stream_right_from_file_and_temp, how='inner', lsuffix='_left', rsuffix='_right')
            if len(df_eda_24hrs_merged.index) > 20:
                eda_24hrs_mean_difference_r_l, eda_24hrs_scrs_diff_r_l, eda_24hrs_scrs_mean_ampl_diff_r_l, eda_24hrs_scl_cvx_mean_diff_r_l, eda_24hrs_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(df_eda_24hrs_merged)

                recording_time_fraction_24hrs_merged = float(
                    len(df_eda_24hrs_merged.index)) / max_24hrs_signal_length
                df_features.ix[start_time, 'recording_time_fraction_24hrs_merged'] = recording_time_fraction_24hrs_merged
                df_features.ix[start_time, 'eda_24hrs_mean_difference_r_l'] = eda_24hrs_mean_difference_r_l
                df_features.ix[start_time, 'eda_24hrs_scrs_diff_r_l'] = eda_24hrs_scrs_diff_r_l
                df_features.ix[start_time, 'eda_24hrs_scrs_mean_ampl_diff_r_l'] = eda_24hrs_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_24hrs_scl_cvx_mean_diff_r_l'] = eda_24hrs_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_24hrs_scr_cvx_mean_diff_r_l'] = eda_24hrs_scr_cvx_mean_diff_r_l



        # percentage time recording
        recording_time_fraction_24hrs_left = float(len(eda_stream_left_from_file_and_temp.index)) / max_24hrs_signal_length
        df_features.ix[start_time, 'recording_time_fraction_24hrs_left'] = recording_time_fraction_24hrs_left
        df_features.ix[start_time, 'eda_24hrs_mean_left'] = eda_24hrs_mean_left
        df_features.ix[start_time, 'eda_24hrs_scrs_left'] = eda_24hrs_scrs_left
        df_features.ix[start_time, 'eda_24hrs_scrs_mean_ampl_left'] = eda_24hrs_scrs_mean_ampl_left
        df_features.ix[start_time, 'eda_24hrs_scl_cvx_mean_left'] = eda_24hrs_scl_cvx_mean_left
        df_features.ix[start_time, 'eda_24hrs_scr_cvx_mean_left'] = eda_24hrs_scr_cvx_mean_left

        # most mobile period of 24hrs: 0-6 night, 6-12 morning, 12-18 afternoon, 18-24 evening
        start_morning_time = start_time.replace(hour=6)
        start_afternoon_time = start_time.replace(hour=12)
        start_evening_time = start_time.replace(hour=18)

        eda_stream_left_from_file_and_temp_night = eda_stream_left_from_file_and_temp.ix[start_time:start_morning_time,]
        eda_stream_left_from_file_and_temp_morning = eda_stream_left_from_file_and_temp.ix[start_morning_time:start_afternoon_time,]
        eda_stream_left_from_file_and_temp_afternoon = eda_stream_left_from_file_and_temp.ix[start_afternoon_time:start_evening_time,]
        eda_stream_left_from_file_and_temp_evening = eda_stream_left_from_file_and_temp.ix[start_evening_time:end_time,]


        # percentage time recording night, morning, afternoon, evening
        recording_time_fraction_morning_left = float(
            len(eda_stream_left_from_file_and_temp_morning.index)) / max_morning_afternoon_evening_night_signal_length
        recording_time_fraction_afternoon_left = float(
            len(eda_stream_left_from_file_and_temp_afternoon.index)) / max_morning_afternoon_evening_night_signal_length
        recording_time_fraction_evening_left = float(
            len(eda_stream_left_from_file_and_temp_evening.index)) / max_morning_afternoon_evening_night_signal_length
        recording_time_fraction_night_left = float(
            len(eda_stream_left_from_file_and_temp_night.index)) / max_morning_afternoon_evening_night_signal_length
        df_features.ix[start_time, 'recording_time_fraction_morning_left'] = recording_time_fraction_morning_left
        df_features.ix[start_time, 'recording_time_fraction_afternoon_left'] = recording_time_fraction_afternoon_left
        df_features.ix[start_time, 'recording_time_fraction_evening_left'] = recording_time_fraction_evening_left
        df_features.ix[start_time, 'recording_time_fraction_night_left'] = recording_time_fraction_night_left

        # calculate morning EDA features
        if len(eda_stream_left_from_file_and_temp_morning.index) > 20:
            eda_morning_mean_left, eda_morning_scrs_left, eda_morning_scrs_mean_ampl_left, eda_morning_scl_cvx_mean_left, eda_morning_scr_cvx_mean_left = calculate_one_hand_eda_features(eda_stream_left_from_file_and_temp_morning)
            df_features.ix[start_time, 'eda_morning_mean_left'] = eda_morning_mean_left
            df_features.ix[start_time, 'eda_morning_scrs_left'] = eda_morning_scrs_left
            df_features.ix[start_time, 'eda_morning_scrs_mean_ampl_left'] = eda_morning_scrs_mean_ampl_left
            df_features.ix[start_time, 'eda_morning_scl_cvx_mean_left'] = eda_morning_scl_cvx_mean_left
            df_features.ix[start_time, 'eda_morning_scr_cvx_mean_left'] = eda_morning_scr_cvx_mean_left

        # calculate afternoon EDA features
        if len(eda_stream_left_from_file_and_temp_afternoon.index) > 20:
            eda_afternoon_mean_left, eda_afternoon_scrs_left, eda_afternoon_scrs_mean_ampl_left, eda_afternoon_scl_cvx_mean_left, eda_afternoon_scr_cvx_mean_left = calculate_one_hand_eda_features(
                eda_stream_left_from_file_and_temp_afternoon)
            df_features.ix[start_time, 'eda_afternoon_mean_left'] = eda_afternoon_mean_left
            df_features.ix[start_time, 'eda_afternoon_scrs_left'] = eda_afternoon_scrs_left
            df_features.ix[start_time, 'eda_afternoon_scrs_mean_ampl_left'] = eda_afternoon_scrs_mean_ampl_left
            df_features.ix[start_time, 'eda_afternoon_scl_cvx_mean_left'] = eda_afternoon_scl_cvx_mean_left
            df_features.ix[start_time, 'eda_afternoon_scr_cvx_mean_left'] = eda_afternoon_scr_cvx_mean_left

        # calculate evening EDA features
        if len(eda_stream_left_from_file_and_temp_evening.index) > 20:
            eda_evening_mean_left, eda_evening_scrs_left, eda_evening_scrs_mean_ampl_left, eda_evening_scl_cvx_mean_left, eda_evening_scr_cvx_mean_left = calculate_one_hand_eda_features(
                eda_stream_left_from_file_and_temp_evening)
            df_features.ix[start_time, 'eda_evening_mean_left'] = eda_evening_mean_left
            df_features.ix[start_time, 'eda_evening_scrs_left'] = eda_evening_scrs_left
            df_features.ix[start_time, 'eda_evening_scrs_mean_ampl_left'] = eda_evening_scrs_mean_ampl_left
            df_features.ix[start_time, 'eda_evening_scl_cvx_mean_left'] = eda_evening_scl_cvx_mean_left
            df_features.ix[start_time, 'eda_evening_scr_cvx_mean_left'] = eda_evening_scr_cvx_mean_left

        # calculate night EDA features
        if len(eda_stream_left_from_file_and_temp_night.index) > 20:
            eda_night_mean_left, eda_night_scrs_left, eda_night_scrs_mean_ampl_left, eda_night_scl_cvx_mean_left, eda_night_scr_cvx_mean_left = calculate_one_hand_eda_features(
                eda_stream_left_from_file_and_temp_night)
            df_features.ix[start_time, 'eda_night_mean_left'] = eda_night_mean_left
            df_features.ix[start_time, 'eda_night_scrs_left'] = eda_night_scrs_left
            df_features.ix[start_time, 'eda_night_scrs_mean_ampl_left'] = eda_night_scrs_mean_ampl_left
            df_features.ix[start_time, 'eda_night_scl_cvx_mean_left'] = eda_night_scl_cvx_mean_left
            df_features.ix[start_time, 'eda_night_scr_cvx_mean_left'] = eda_night_scr_cvx_mean_left



        # calculate eda features for the merged (R vs L) strem
        if len(eda_stream_right_from_file_and_temp.index) > 20:
            df_eda_night_merged = eda_stream_left_from_file_and_temp_night.join(eda_stream_right_from_file_and_temp, how='inner', lsuffix='_left', rsuffix='_right')
            df_eda_morning_merged = eda_stream_left_from_file_and_temp_morning.join(eda_stream_right_from_file_and_temp, how='inner', lsuffix='_left', rsuffix='_right')
            df_eda_afternoon_merged = eda_stream_left_from_file_and_temp_afternoon.join(eda_stream_right_from_file_and_temp, how='inner', lsuffix='_left', rsuffix='_right')
            df_eda_evening_merged = eda_stream_left_from_file_and_temp_evening.join(eda_stream_right_from_file_and_temp, how='inner', lsuffix='_left', rsuffix='_right')

            if len(df_eda_morning_merged.index) > 20:
                eda_morning_mean_difference_r_l, eda_morning_scrs_diff_r_l, eda_morning_scrs_mean_ampl_diff_r_l, eda_morning_scl_cvx_mean_diff_r_l, eda_morning_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                    df_eda_morning_merged)

                recording_time_fraction_morning_merged = float(
                    len(df_eda_morning_merged.index)) / max_morning_afternoon_evening_night_signal_length
                df_features.ix[start_time, 'recording_time_fraction_morning_merged'] = recording_time_fraction_morning_merged
                df_features.ix[start_time, 'eda_morning_mean_difference_r_l'] = eda_morning_mean_difference_r_l
                df_features.ix[start_time, 'eda_morning_scrs_diff_r_l'] = eda_morning_scrs_diff_r_l
                df_features.ix[start_time, 'eda_morning_scrs_mean_ampl_diff_r_l'] = eda_morning_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_morning_scl_cvx_mean_diff_r_l'] = eda_morning_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_morning_scr_cvx_mean_diff_r_l'] = eda_morning_scr_cvx_mean_diff_r_l

            if len(df_eda_afternoon_merged.index) > 20:
                eda_afternoon_mean_difference_r_l, eda_afternoon_scrs_diff_r_l, eda_afternoon_scrs_mean_ampl_diff_r_l, eda_afternoon_scl_cvx_mean_diff_r_l, eda_afternoon_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                    df_eda_afternoon_merged)

                recording_time_fraction_afternoon_merged = float(
                    len(df_eda_afternoon_merged.index)) / max_morning_afternoon_evening_night_signal_length
                df_features.ix[start_time, 'recording_time_fraction_afternoon_merged'] = recording_time_fraction_afternoon_merged
                df_features.ix[start_time, 'eda_afternoon_mean_difference_r_l'] = eda_afternoon_mean_difference_r_l
                df_features.ix[start_time, 'eda_afternoon_scrs_diff_r_l'] = eda_afternoon_scrs_diff_r_l
                df_features.ix[start_time, 'eda_afternoon_scrs_mean_ampl_diff_r_l'] = eda_afternoon_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_afternoon_scl_cvx_mean_diff_r_l'] = eda_afternoon_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_afternoon_scr_cvx_mean_diff_r_l'] = eda_afternoon_scr_cvx_mean_diff_r_l

            if len(df_eda_evening_merged.index) > 20:
                eda_evening_mean_difference_r_l, eda_evening_scrs_diff_r_l, eda_evening_scrs_mean_ampl_diff_r_l, eda_evening_scl_cvx_mean_diff_r_l, eda_evening_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                    df_eda_evening_merged)

                recording_time_fraction_evening_merged = float(
                    len(df_eda_evening_merged.index)) / max_morning_afternoon_evening_night_signal_length
                df_features.ix[start_time, 'recording_time_fraction_evening_merged'] = recording_time_fraction_evening_merged
                df_features.ix[start_time, 'eda_evening_mean_difference_r_l'] = eda_evening_mean_difference_r_l
                df_features.ix[start_time, 'eda_evening_scrs_diff_r_l'] = eda_evening_scrs_diff_r_l
                df_features.ix[start_time, 'eda_evening_scrs_mean_ampl_diff_r_l'] = eda_evening_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_evening_scl_cvx_mean_diff_r_l'] = eda_evening_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_evening_scr_cvx_mean_diff_r_l'] = eda_evening_scr_cvx_mean_diff_r_l

            if len(df_eda_night_merged.index) > 20:
                eda_night_mean_difference_r_l, eda_night_scrs_diff_r_l, eda_night_scrs_mean_ampl_diff_r_l, eda_night_scl_cvx_mean_diff_r_l, eda_night_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                    df_eda_night_merged)

                recording_time_fraction_night_merged = float(
                    len(df_eda_night_merged.index)) / max_morning_afternoon_evening_night_signal_length
                df_features.ix[start_time, 'recording_time_fraction_night_merged'] = recording_time_fraction_night_merged
                df_features.ix[start_time, 'eda_night_mean_difference_r_l'] = eda_night_mean_difference_r_l
                df_features.ix[start_time, 'eda_night_scrs_diff_r_l'] = eda_night_scrs_diff_r_l
                df_features.ix[start_time, 'eda_night_scrs_mean_ampl_diff_r_l'] = eda_night_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_night_scl_cvx_mean_diff_r_l'] = eda_night_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_night_scr_cvx_mean_diff_r_l'] = eda_night_scr_cvx_mean_diff_r_l

    # calculate eda features for right hand stream
    if len(eda_stream_right_from_file_and_temp.index) < 21:
        print ('There is no recording from Right hand for the specified time interval!')
    else:
        # calculate 24hrs EDA features
        eda_24hrs_mean_right, eda_24hrs_scrs_right, eda_24hrs_scrs_mean_ampl_right, eda_24hrs_scl_cvx_mean_right, \
            eda_24hrs_scr_cvx_mean_right = calculate_one_hand_eda_features(eda_stream_right_from_file_and_temp)

        # percentage time recording
        recording_time_fraction_24hrs_right = float(
            len(eda_stream_right_from_file_and_temp.index)) / max_24hrs_signal_length
        df_features.ix[start_time, 'recording_time_fraction_24hrs_right'] = recording_time_fraction_24hrs_right
        df_features.ix[start_time, 'eda_24hrs_mean_right'] = eda_24hrs_mean_right
        df_features.ix[start_time, 'eda_24hrs_scrs_right'] = eda_24hrs_scrs_right
        df_features.ix[start_time, 'eda_24hrs_scrs_mean_ampl_right'] = eda_24hrs_scrs_mean_ampl_right
        df_features.ix[start_time, 'eda_24hrs_scl_cvx_mean_right'] = eda_24hrs_scl_cvx_mean_right
        df_features.ix[start_time, 'eda_24hrs_scr_cvx_mean_right'] = eda_24hrs_scr_cvx_mean_right

        # most mobile period of 24hrs: 0-6 night, 6-12 morning, 12-18 afternoon, 18-24 evening
        start_morning_time = start_time.replace(hour=6)
        start_afternoon_time = start_time.replace(hour=12)
        start_evening_time = start_time.replace(hour=18)

        eda_stream_right_from_file_and_temp_night = eda_stream_right_from_file_and_temp.ix[
                                                   start_time:start_morning_time, ]
        eda_stream_right_from_file_and_temp_morning = eda_stream_right_from_file_and_temp.ix[
                                                     start_morning_time:start_afternoon_time, ]
        eda_stream_right_from_file_and_temp_afternoon = eda_stream_right_from_file_and_temp.ix[
                                                       start_afternoon_time:start_evening_time, ]
        eda_stream_right_from_file_and_temp_evening = eda_stream_right_from_file_and_temp.ix[
                                                     start_evening_time:end_time, ]

        # percentage time recording night, morning, afternoon, evening
        recording_time_fraction_morning_right = float(
            len(eda_stream_right_from_file_and_temp_morning.index)) / max_morning_afternoon_evening_night_signal_length
        recording_time_fraction_afternoon_right = float(
            len(eda_stream_right_from_file_and_temp_afternoon.index)) / max_morning_afternoon_evening_night_signal_length
        recording_time_fraction_evening_right = float(
            len(eda_stream_right_from_file_and_temp_evening.index)) / max_morning_afternoon_evening_night_signal_length
        recording_time_fraction_night_right = float(
            len(eda_stream_right_from_file_and_temp_night.index)) / max_morning_afternoon_evening_night_signal_length
        df_features.ix[start_time, 'recording_time_fraction_morning_right'] = recording_time_fraction_morning_right
        df_features.ix[start_time, 'recording_time_fraction_afternoon_right'] = recording_time_fraction_afternoon_right
        df_features.ix[start_time, 'recording_time_fraction_evening_right'] = recording_time_fraction_evening_right
        df_features.ix[start_time, 'recording_time_fraction_night_right'] = recording_time_fraction_night_right

        # calculate morning EDA features
        if len(eda_stream_right_from_file_and_temp_morning.index) > 20:
            eda_morning_mean_right, eda_morning_scrs_right, eda_morning_scrs_mean_ampl_right, eda_morning_scl_cvx_mean_right, eda_morning_scr_cvx_mean_right = calculate_one_hand_eda_features(
                eda_stream_right_from_file_and_temp_morning)
            df_features.ix[start_time, 'eda_morning_mean_right'] = eda_morning_mean_right
            df_features.ix[start_time, 'eda_morning_scrs_right'] = eda_morning_scrs_right
            df_features.ix[start_time, 'eda_morning_scrs_mean_ampl_right'] = eda_morning_scrs_mean_ampl_right
            df_features.ix[start_time, 'eda_morning_scl_cvx_mean_right'] = eda_morning_scl_cvx_mean_right
            df_features.ix[start_time, 'eda_morning_scr_cvx_mean_right'] = eda_morning_scr_cvx_mean_right

        # calculate afternoon EDA features
        if len(eda_stream_right_from_file_and_temp_afternoon.index) > 20:
            eda_afternoon_mean_right, eda_afternoon_scrs_right, eda_afternoon_scrs_mean_ampl_right, eda_afternoon_scl_cvx_mean_right, eda_afternoon_scr_cvx_mean_right = calculate_one_hand_eda_features(
                eda_stream_right_from_file_and_temp_afternoon)
            df_features.ix[start_time, 'eda_afternoon_mean_right'] = eda_afternoon_mean_right
            df_features.ix[start_time, 'eda_afternoon_scrs_right'] = eda_afternoon_scrs_right
            df_features.ix[start_time, 'eda_afternoon_scrs_mean_ampl_right'] = eda_afternoon_scrs_mean_ampl_right
            df_features.ix[start_time, 'eda_afternoon_scl_cvx_mean_right'] = eda_afternoon_scl_cvx_mean_right
            df_features.ix[start_time, 'eda_afternoon_scr_cvx_mean_right'] = eda_afternoon_scr_cvx_mean_right

        # calculate evening EDA features
        if len(eda_stream_right_from_file_and_temp_evening.index) > 20:
            eda_evening_mean_right, eda_evening_scrs_right, eda_evening_scrs_mean_ampl_right, eda_evening_scl_cvx_mean_right, eda_evening_scr_cvx_mean_right = calculate_one_hand_eda_features(
                eda_stream_right_from_file_and_temp_evening)
            df_features.ix[start_time, 'eda_evening_mean_right'] = eda_evening_mean_right
            df_features.ix[start_time, 'eda_evening_scrs_right'] = eda_evening_scrs_right
            df_features.ix[start_time, 'eda_evening_scrs_mean_ampl_right'] = eda_evening_scrs_mean_ampl_right
            df_features.ix[start_time, 'eda_evening_scl_cvx_mean_right'] = eda_evening_scl_cvx_mean_right
            df_features.ix[start_time, 'eda_evening_scr_cvx_mean_right'] = eda_evening_scr_cvx_mean_right

        # calculate night EDA features
        if len(eda_stream_right_from_file_and_temp_night.index) > 20:
            eda_night_mean_right, eda_night_scrs_right, eda_night_scrs_mean_ampl_right, eda_night_scl_cvx_mean_right, eda_night_scr_cvx_mean_right = calculate_one_hand_eda_features(
                eda_stream_right_from_file_and_temp_night)
            df_features.ix[start_time, 'eda_night_mean_right'] = eda_night_mean_right
            df_features.ix[start_time, 'eda_night_scrs_right'] = eda_night_scrs_right
            df_features.ix[start_time, 'eda_night_scrs_mean_ampl_right'] = eda_night_scrs_mean_ampl_right
            df_features.ix[start_time, 'eda_night_scl_cvx_mean_right'] = eda_night_scl_cvx_mean_right
            df_features.ix[start_time, 'eda_night_scr_cvx_mean_right'] = eda_night_scr_cvx_mean_right

        return df_features

def calculate_both_hands_eda_features(df_eda_merged):

        # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:4hz, order:6)
        df_eda_merged['filtered_eda_left'] = butter_lowpass_filter(df_eda_merged['eda_left'], 1.0, fs_eda_e4, 6)
        df_eda_merged['filtered_eda_right'] = butter_lowpass_filter(df_eda_merged['eda_right'], 1.0, fs_eda_e4, 6)

        # average SCL L & R
        eda_mean_difference_r_l = df_eda_merged.filtered_eda_right.mean() - df_eda_merged.filtered_eda_left.mean()

        # SCRs
        ts_l, filtered_eda_l, scr_onset_indices_l, peaks_indices_l, peak_amplitudes_l = eda.eda(
            signal=df_eda_merged['eda_left'].values.flatten(),
            sampling_rate=fs_eda_e4, show=False)
        eda_scrs_l = len(scr_onset_indices_l)
        eda_scrs_mean_ampl_l = np.mean(peak_amplitudes_l)

        ts_r, filtered_eda_r, scr_onset_indices_r, peaks_indices_r, peak_amplitudes_r = eda.eda(
            signal=df_eda_merged['eda_right'].values.flatten(),
            sampling_rate=fs_eda_e4, show=False)
        eda_scrs_r = len(scr_onset_indices_r)
        eda_scrs_mean_ampl_r = np.mean(peak_amplitudes_r)

        eda_scrs_diff_r_l = eda_scrs_r - eda_scrs_l
        eda_scrs_mean_ampl_diff_r_l = eda_scrs_mean_ampl_r - eda_scrs_mean_ampl_l



        # cvx-based calculation
        y_l = df_eda_merged.eda_left.values
        yn_l = (y_l - y_l.mean()) / y_l.std()
        Fs = float(fs_eda_e4)
        [r_l, p_l, t_l, l_l, d_l, e_l, obj_l] = cvxEDA.cvxEDA(yn_l, 1. / Fs)

        y_r = df_eda_merged.eda_right.values
        yn_r = (y_r - y_r.mean()) / y_r.std()
        Fs = float(fs_eda_e4)
        [r_r, p_r, t_r, l_r, d_r, e_r, obj_r] = cvxEDA.cvxEDA(yn_r, 1. / Fs)

        eda_scl_cvx_mean_diff_r_l = t_r.mean() - t_l.mean()
        eda_scr_cvx_mean_diff_r_l = r_r.mean() - r_l.mean()

        return eda_mean_difference_r_l, eda_scrs_diff_r_l, eda_scrs_mean_ampl_diff_r_l, eda_scl_cvx_mean_diff_r_l, eda_scr_cvx_mean_diff_r_l

def calculate_one_hand_eda_features(df_eda):
    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:4hz, order:6)
    df_eda['filtered_eda'] =  butter_lowpass_filter(df_eda['eda'], 1.0, fs_eda_e4, 6)

    # average SCL L & R
    eda_mean = df_eda.filtered_eda.mean()

    # SCRs
    ts, filtered_eda, scr_onset_indices, peaks_indices, peak_amplitudes = eda.eda(
        signal=df_eda['eda'].values.flatten(),
        sampling_rate=fs_eda_e4, show=False)
    eda_scrs = len(scr_onset_indices)
    eda_scrs_mean_ampl = np.mean(peak_amplitudes)
    #cvx-based calculation
    y = df_eda.eda.values
    yn = (y - y.mean()) / y.std()
    Fs = float(fs_eda_e4)
    [r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1. / Fs)
    eda_scl_cvx_mean = t.mean()
    eda_scr_cvx_mean = r.mean()
    return eda_mean, eda_scrs, eda_scrs_mean_ampl, eda_scl_cvx_mean, eda_scr_cvx_mean


def calculate_24hrs_eda_features():
    print (user_id)
    eda_first_row_left = pd.read_hdf(eda_filepath, 'EDA_left', start=0, stop=1)
    start_date_left = eda_first_row_left.index[0]
    eda_last_row_left = pd.read_hdf(eda_filepath, 'EDA_left', start=-2, stop=-1)
    end_date_left = eda_last_row_left.index[0]  # for final version remove that line and uncomment previous line

    eda_first_row_right = pd.read_hdf(eda_filepath, 'EDA_right', start=0, stop=1)
    start_date_right = eda_first_row_right.index[0]
    eda_last_row_right = pd.read_hdf(eda_filepath, 'EDA_right', start=-2, stop=-1)
    end_date_right = eda_last_row_right.index[0]

    start_date = min(start_date_left, start_date_right)
    start_date = start_date.replace(hour=0, minute=0, second=0)
    end_date = max(end_date_left, end_date_right)
    end_date = end_date.replace(hour=0, minute=0, second=0)
    rng = pd.date_range(start_date, end_date)

    df_eda_features = pd.DataFrame()
    for idx, beginning in enumerate(rng):
        # if idx != 35 & idx != 38:
        if idx > 38:
            print (idx, beginning)
            end = beginning + timedelta(hours=24)

            df_24hrs_features = read_hdfs_and_calculate_features(acc_filepath, eda_filepath, temp_filepath, beginning, end)
            df_eda_features = df_eda_features.append(df_24hrs_features)
            # if (df_24hrs_features is not None):
            #     with open(eda_features_filepath, 'a') as f:
            #         df_24hrs_features.to_csv(f, header=True)  # for the blank file change header to True, to write column names

    save_features = input('Do you want to save the calculated features (y/n): ') #raw_input in python 2
    if (save_features == 'y'):
        with open(eda_features_filepath, 'a') as f:
            df_eda_features.to_csv(f, header=True)  # for the blank file change header to True, to write column names

def read_eda_acc_hdfs_and_calculate_features(a_filepath, e_filepath, t_path, start_time, end_time):
    has_enought_recording_from_left_to_process = False
    has_enought_recording_from_right_to_process = False


    acc_stream_left_from_file = pd.read_hdf(a_filepath, 'ACC_left',
                                            where='index>=start_time & index<end_time & columns = z')
    print ('Finished reading Left Acc')
    acc_stream_right_from_file = pd.read_hdf(a_filepath, 'ACC_right',
                                             where='index>=start_time & index<end_time & columns = z')
    print ('Finished reading Right Acc')

    eda_stream_left_from_file = pd.read_hdf(e_filepath, 'EDA_left',
                                            where='index>=start_time & index<end_time')
    print ('Finished reading Left Eda')

    eda_stream_right_from_file = pd.read_hdf(e_filepath, 'EDA_right',
                                             where='index>=start_time & index<end_time')
    print ('Finished reading Right EDA')

    temp_stream_left_from_file = pd.read_hdf(t_path, 'TEMP_left',
                                            where='index>=start_time & index<end_time')
    print ('Finished reading Left TEMP')
    temp_stream_right_from_file = pd.read_hdf(t_path, 'TEMP_right',
                                             where='index>=start_time & index<end_time')
    print ('Finished reading Right TEMP')

    eda_stream_left_from_file_and_temp = eda_stream_left_from_file.join(temp_stream_left_from_file, how='left')
    eda_stream_right_from_file_and_temp = eda_stream_right_from_file.join(temp_stream_right_from_file, how='left')

    if filter_out_based_on_temp_threshold:
        eda_stream_left_from_file_and_temp = eda_stream_left_from_file_and_temp[eda_stream_left_from_file_and_temp.temp > min_valid_temp_value]
        eda_stream_right_from_file_and_temp = eda_stream_right_from_file[eda_stream_right_from_file_and_temp.temp > min_valid_temp_value]

    df_features = pd.DataFrame( )
    df_features.ix[start_time, 'ID'] = user_id

    max_24hrs_signal_length = 24 * 60 * 60 * fs_eda_e4
    max_morning_afternoon_evening_night_signal_length = 6 * 60 * 60 * fs_eda_e4 - 1


    if len(acc_stream_left_from_file.index) == 0:
        print ('There is no recording from Left hand for the specified time interval!')
    else:
        # adapt the scale [-2g, 2g]
        acc_stream_left_from_file = acc_stream_left_from_file.div(64)
        acc_stream_empatica_vm_left = generate_empatica_motion_variable_from_y_fast(acc_stream_left_from_file)

        eda_acc_vm_stream_left_combined = eda_stream_left_from_file_and_temp.join(acc_stream_empatica_vm_left, how='left')

        eda_acc_vm_stream_left_combined['user_still'] = [0] * len(eda_acc_vm_stream_left_combined.index)
        eda_acc_vm_stream_left_combined.user_still[eda_acc_vm_stream_left_combined.empatica_motion_vector < active_motion_threshold] = 1
        eda_acc_vm_stream_left_combined['motionless_segments_length'] = eda_acc_vm_stream_left_combined.ix[::-1, :].groupby((eda_acc_vm_stream_left_combined.ix[::-1, 'user_still'] == 0).cumsum()).cumcount()[::-1]

        eda_acc_vm_stream_left_combined['activity_change'] = eda_acc_vm_stream_left_combined.user_still.diff()
        eda_acc_vm_stream_left_combined.ix[0, 'activity_change'] = eda_acc_vm_stream_left_combined.ix[0, 'user_still']

        eda_acc_vm_stream_left_combined['valid_motionless_eda_segments'] = np.nan * len(eda_acc_vm_stream_left_combined.index)
        eda_acc_vm_stream_left_combined.ix[eda_acc_vm_stream_left_combined.activity_change == 1, 'valid_motionless_eda_segments'] = eda_acc_vm_stream_left_combined.motionless_segments_length
        eda_acc_vm_stream_left_combined.ix[eda_acc_vm_stream_left_combined.activity_change == -1, 'valid_motionless_eda_segments'] = eda_acc_vm_stream_left_combined.activity_change
        eda_acc_vm_stream_left_combined.ix[:, 'valid_motionless_eda_segments'] = eda_acc_vm_stream_left_combined.ix[:, 'valid_motionless_eda_segments'].fillna(method='ffill')
        df_eda_acc_vm_stream_left_combined_motionless = eda_acc_vm_stream_left_combined.ix[eda_acc_vm_stream_left_combined.valid_motionless_eda_segments > min_motionless_eda_segment_len, :]

        if len(df_eda_acc_vm_stream_left_combined_motionless.index) < 21:
            print ('There is not enough motionless datapoints from the Left hand in a 24hrs interval')
        else:
            has_enought_recording_from_left_to_process = True
            #calculate 24hrs left hand features
            eda_24hrs_mean_left, eda_24hrs_scrs_left, eda_24hrs_scrs_mean_ampl_left, eda_24hrs_scl_cvx_mean_left, \
            eda_24hrs_scr_cvx_mean_left = calculate_one_hand_eda_features(df_eda_acc_vm_stream_left_combined_motionless)

            # percentage time recording
            recording_time_fraction_24hrs_left = float(
                len(df_eda_acc_vm_stream_left_combined_motionless.index)) / max_24hrs_signal_length
            df_features.ix[start_time, 'recording_time_fraction_24hrs_left_motionless'] = recording_time_fraction_24hrs_left
            df_features.ix[start_time, 'eda_motionless_24hrs_mean_left'] = eda_24hrs_mean_left
            df_features.ix[start_time, 'eda_motionless_24hrs_scrs_left'] = eda_24hrs_scrs_left
            df_features.ix[start_time, 'eda_motionless_24hrs_scrs_mean_ampl_left'] = eda_24hrs_scrs_mean_ampl_left
            df_features.ix[start_time, 'eda_motionless_24hrs_scl_cvx_mean_left'] = eda_24hrs_scl_cvx_mean_left
            df_features.ix[start_time, 'eda_motionless_24hrs_scr_cvx_mean_left'] = eda_24hrs_scr_cvx_mean_left

            # most mobile period of 24hrs: 0-6 night, 6-12 morning, 12-18 afternoon, 18-24 evening
            start_morning_time = start_time.replace(hour=6)
            start_afternoon_time = start_time.replace(hour=12)
            start_evening_time = start_time.replace(hour=18)

            eda_stream_left_from_file_and_temp_night = df_eda_acc_vm_stream_left_combined_motionless.ix[
                                                       start_time:start_morning_time, ]
            eda_stream_left_from_file_and_temp_morning = df_eda_acc_vm_stream_left_combined_motionless.ix[
                                                         start_morning_time:start_afternoon_time, ]
            eda_stream_left_from_file_and_temp_afternoon = df_eda_acc_vm_stream_left_combined_motionless.ix[
                                                           start_afternoon_time:start_evening_time, ]
            eda_stream_left_from_file_and_temp_evening = df_eda_acc_vm_stream_left_combined_motionless.ix[
                                                         start_evening_time:end_time, ]

            # percentage time recording night, morning, afternoon, evening
            recording_time_fraction_morning_left = float(
                len(
                    eda_stream_left_from_file_and_temp_morning.index)) / max_morning_afternoon_evening_night_signal_length
            recording_time_fraction_afternoon_left = float(
                len(
                    eda_stream_left_from_file_and_temp_afternoon.index)) / max_morning_afternoon_evening_night_signal_length
            recording_time_fraction_evening_left = float(
                len(
                    eda_stream_left_from_file_and_temp_evening.index)) / max_morning_afternoon_evening_night_signal_length
            recording_time_fraction_night_left = float(
                len(eda_stream_left_from_file_and_temp_night.index)) / max_morning_afternoon_evening_night_signal_length
            df_features.ix[start_time, 'recording_time_fraction_morning_left_motionless'] = recording_time_fraction_morning_left
            df_features.ix[
                start_time, 'recording_time_fraction_afternoon_left_motionless'] = recording_time_fraction_afternoon_left
            df_features.ix[start_time, 'recording_time_fraction_evening_left_motionless'] = recording_time_fraction_evening_left
            df_features.ix[start_time, 'recording_time_fraction_night_left_motionless'] = recording_time_fraction_night_left

            # calculate morning EDA features
            if len(eda_stream_left_from_file_and_temp_morning.index) > 20:
                eda_morning_mean_left, eda_morning_scrs_left, eda_morning_scrs_mean_ampl_left, eda_morning_scl_cvx_mean_left, eda_morning_scr_cvx_mean_left = calculate_one_hand_eda_features(
                    eda_stream_left_from_file_and_temp_morning)
                df_features.ix[start_time, 'eda_motionless_morning_mean_left'] = eda_morning_mean_left
                df_features.ix[start_time, 'eda_motionless_morning_scrs_left'] = eda_morning_scrs_left
                df_features.ix[start_time, 'eda_motionless_morning_scrs_mean_ampl_left'] = eda_morning_scrs_mean_ampl_left
                df_features.ix[start_time, 'eda_motionless_morning_scl_cvx_mean_left'] = eda_morning_scl_cvx_mean_left
                df_features.ix[start_time, 'eda_motionless_morning_scr_cvx_mean_left'] = eda_morning_scr_cvx_mean_left

            # calculate afternoon EDA features
            if len(eda_stream_left_from_file_and_temp_afternoon.index) > 20:
                eda_afternoon_mean_left, eda_afternoon_scrs_left, eda_afternoon_scrs_mean_ampl_left, eda_afternoon_scl_cvx_mean_left, eda_afternoon_scr_cvx_mean_left = calculate_one_hand_eda_features(
                    eda_stream_left_from_file_and_temp_afternoon)
                df_features.ix[start_time, 'eda_motionless_afternoon_mean_left'] = eda_afternoon_mean_left
                df_features.ix[start_time, 'eda_motionless_afternoon_scrs_left'] = eda_afternoon_scrs_left
                df_features.ix[start_time, 'eda_motionless_afternoon_scrs_mean_ampl_left'] = eda_afternoon_scrs_mean_ampl_left
                df_features.ix[start_time, 'eda_motionless_afternoon_scl_cvx_mean_left'] = eda_afternoon_scl_cvx_mean_left
                df_features.ix[start_time, 'eda_motionless_afternoon_scr_cvx_mean_left'] = eda_afternoon_scr_cvx_mean_left

            # calculate evening EDA features
            if len(eda_stream_left_from_file_and_temp_evening.index) > 20:
                eda_evening_mean_left, eda_evening_scrs_left, eda_evening_scrs_mean_ampl_left, eda_evening_scl_cvx_mean_left, eda_evening_scr_cvx_mean_left = calculate_one_hand_eda_features(
                    eda_stream_left_from_file_and_temp_evening)
                df_features.ix[start_time, 'eda_motionless_evening_mean_left'] = eda_evening_mean_left
                df_features.ix[start_time, 'eda_motionless_evening_scrs_left'] = eda_evening_scrs_left
                df_features.ix[start_time, 'eda_motionless_evening_scrs_mean_ampl_left'] = eda_evening_scrs_mean_ampl_left
                df_features.ix[start_time, 'eda_motionless_evening_scl_cvx_mean_left'] = eda_evening_scl_cvx_mean_left
                df_features.ix[start_time, 'eda_motionless_evening_scr_cvx_mean_left'] = eda_evening_scr_cvx_mean_left

            # calculate night EDA features
            if len(eda_stream_left_from_file_and_temp_night.index) > 20:
                eda_night_mean_left, eda_night_scrs_left, eda_night_scrs_mean_ampl_left, eda_night_scl_cvx_mean_left, eda_night_scr_cvx_mean_left = calculate_one_hand_eda_features(
                    eda_stream_left_from_file_and_temp_night)
                df_features.ix[start_time, 'eda_motionless_night_mean_left'] = eda_night_mean_left
                df_features.ix[start_time, 'eda_motionless_night_scrs_left'] = eda_night_scrs_left
                df_features.ix[start_time, 'eda_motionless_night_scrs_mean_ampl_left'] = eda_night_scrs_mean_ampl_left
                df_features.ix[start_time, 'eda_motionless_night_scl_cvx_mean_left'] = eda_night_scl_cvx_mean_left
                df_features.ix[start_time, 'eda_motionless_night_scr_cvx_mean_left'] = eda_night_scr_cvx_mean_left




    if len(acc_stream_right_from_file.index) == 0:
        print ('There is no recording from Right hand for the specified time interval!')
    else:
        # adapt the scale [-2g, 2g]
        acc_stream_right_from_file = acc_stream_right_from_file.div(64)
        acc_stream_empatica_vm_right = generate_empatica_motion_variable_from_y_fast(acc_stream_right_from_file)

        eda_acc_vm_stream_right_combined = eda_stream_right_from_file_and_temp.join(acc_stream_empatica_vm_right, how='left')

        eda_acc_vm_stream_right_combined['user_still'] = [0] * len(eda_acc_vm_stream_right_combined.index)
        eda_acc_vm_stream_right_combined.user_still[eda_acc_vm_stream_right_combined.empatica_motion_vector < active_motion_threshold] = 1
        eda_acc_vm_stream_right_combined['motionless_segments_length'] = eda_acc_vm_stream_right_combined.ix[::-1, :].groupby((eda_acc_vm_stream_right_combined.ix[::-1, 'user_still'] == 0).cumsum()).cumcount()[::-1]

        eda_acc_vm_stream_right_combined['activity_change'] = eda_acc_vm_stream_right_combined.user_still.diff()
        eda_acc_vm_stream_right_combined.ix[0, 'activity_change'] = eda_acc_vm_stream_right_combined.ix[0, 'user_still']

        eda_acc_vm_stream_right_combined['valid_motionless_eda_segments'] = np.nan * len(eda_acc_vm_stream_right_combined.index)
        eda_acc_vm_stream_right_combined.ix[eda_acc_vm_stream_right_combined.activity_change == 1, 'valid_motionless_eda_segments'] = eda_acc_vm_stream_right_combined.motionless_segments_length
        eda_acc_vm_stream_right_combined.ix[eda_acc_vm_stream_right_combined.activity_change == -1, 'valid_motionless_eda_segments'] = eda_acc_vm_stream_right_combined.activity_change
        eda_acc_vm_stream_right_combined.ix[:, 'valid_motionless_eda_segments'] = eda_acc_vm_stream_right_combined.ix[:, 'valid_motionless_eda_segments'].fillna(method='ffill')
        df_eda_acc_vm_stream_right_combined_motionless = eda_acc_vm_stream_right_combined.ix[eda_acc_vm_stream_right_combined.valid_motionless_eda_segments > min_motionless_eda_segment_len, :]

        if len(df_eda_acc_vm_stream_right_combined_motionless.index) < 21:
            print ('There is not enough motionless datapoints from the Right hand in a 24hrs interval')
        else:
            has_enought_recording_from_right_to_process = True
            #calculate 24hrs left hand features
            eda_24hrs_mean_right, eda_24hrs_scrs_right, eda_24hrs_scrs_mean_ampl_right, eda_24hrs_scl_cvx_mean_right, \
            eda_24hrs_scr_cvx_mean_right = calculate_one_hand_eda_features(df_eda_acc_vm_stream_right_combined_motionless)

            # percentage time recording
            recording_time_fraction_24hrs_right = float(
                len(df_eda_acc_vm_stream_right_combined_motionless.index)) / max_24hrs_signal_length
            df_features.ix[start_time, 'recording_time_fraction_24hrs_right_motionless'] = recording_time_fraction_24hrs_right
            df_features.ix[start_time, 'eda_motionless_24hrs_mean_right'] = eda_24hrs_mean_right
            df_features.ix[start_time, 'eda_motionless_24hrs_scrs_right'] = eda_24hrs_scrs_right
            df_features.ix[start_time, 'eda_motionless_24hrs_scrs_mean_ampl_right'] = eda_24hrs_scrs_mean_ampl_right
            df_features.ix[start_time, 'eda_motionless_24hrs_scl_cvx_mean_right'] = eda_24hrs_scl_cvx_mean_right
            df_features.ix[start_time, 'eda_motionless_24hrs_scr_cvx_mean_right'] = eda_24hrs_scr_cvx_mean_right

            # most mobile period of 24hrs: 0-6 night, 6-12 morning, 12-18 afternoon, 18-24 evening
            start_morning_time = start_time.replace(hour=6)
            start_afternoon_time = start_time.replace(hour=12)
            start_evening_time = start_time.replace(hour=18)

            eda_stream_right_from_file_and_temp_night = df_eda_acc_vm_stream_right_combined_motionless.ix[
                                                       start_time:start_morning_time, ]
            eda_stream_right_from_file_and_temp_morning = df_eda_acc_vm_stream_right_combined_motionless.ix[
                                                         start_morning_time:start_afternoon_time, ]
            eda_stream_right_from_file_and_temp_afternoon = df_eda_acc_vm_stream_right_combined_motionless.ix[
                                                           start_afternoon_time:start_evening_time, ]
            eda_stream_right_from_file_and_temp_evening = df_eda_acc_vm_stream_right_combined_motionless.ix[
                                                         start_evening_time:end_time, ]

            # percentage time recording night, morning, afternoon, evening
            recording_time_fraction_morning_right = float(
                len(
                    eda_stream_right_from_file_and_temp_morning.index)) / max_morning_afternoon_evening_night_signal_length
            recording_time_fraction_afternoon_right = float(
                len(
                    eda_stream_right_from_file_and_temp_afternoon.index)) / max_morning_afternoon_evening_night_signal_length
            recording_time_fraction_evening_right = float(
                len(
                    eda_stream_right_from_file_and_temp_evening.index)) / max_morning_afternoon_evening_night_signal_length
            recording_time_fraction_night_right = float(
                len(eda_stream_right_from_file_and_temp_night.index)) / max_morning_afternoon_evening_night_signal_length
            df_features.ix[start_time, 'recording_time_fraction_morning_right_motionless'] = recording_time_fraction_morning_right
            df_features.ix[
                start_time, 'recording_time_fraction_afternoon_right_motionless'] = recording_time_fraction_afternoon_right
            df_features.ix[start_time, 'recording_time_fraction_evening_right_motionless'] = recording_time_fraction_evening_right
            df_features.ix[start_time, 'recording_time_fraction_night_right_motionless'] = recording_time_fraction_night_right

            # calculate morning EDA features
            if len(eda_stream_right_from_file_and_temp_morning.index) > 20:
                eda_morning_mean_right, eda_morning_scrs_right, eda_morning_scrs_mean_ampl_right, eda_morning_scl_cvx_mean_right, eda_morning_scr_cvx_mean_right = calculate_one_hand_eda_features(
                    eda_stream_right_from_file_and_temp_morning)
                df_features.ix[start_time, 'eda_motionless_morning_mean_right'] = eda_morning_mean_right
                df_features.ix[start_time, 'eda_motionless_morning_scrs_right'] = eda_morning_scrs_right
                df_features.ix[start_time, 'eda_motionless_morning_scrs_mean_ampl_right'] = eda_morning_scrs_mean_ampl_right
                df_features.ix[start_time, 'eda_motionless_morning_scl_cvx_mean_right'] = eda_morning_scl_cvx_mean_right
                df_features.ix[start_time, 'eda_motionless_morning_scr_cvx_mean_right'] = eda_morning_scr_cvx_mean_right

            # calculate afternoon EDA features
            if len(eda_stream_right_from_file_and_temp_afternoon.index) > 20:
                eda_afternoon_mean_right, eda_afternoon_scrs_right, eda_afternoon_scrs_mean_ampl_right, eda_afternoon_scl_cvx_mean_right, eda_afternoon_scr_cvx_mean_right = calculate_one_hand_eda_features(
                    eda_stream_right_from_file_and_temp_afternoon)
                df_features.ix[start_time, 'eda_motionless_afternoon_mean_right'] = eda_afternoon_mean_right
                df_features.ix[start_time, 'eda_motionless_afternoon_scrs_right'] = eda_afternoon_scrs_right
                df_features.ix[start_time, 'eda_motionless_afternoon_scrs_mean_ampl_right'] = eda_afternoon_scrs_mean_ampl_right
                df_features.ix[start_time, 'eda_motionless_afternoon_scl_cvx_mean_right'] = eda_afternoon_scl_cvx_mean_right
                df_features.ix[start_time, 'eda_motionless_afternoon_scr_cvx_mean_right'] = eda_afternoon_scr_cvx_mean_right

            # calculate evening EDA features
            if len(eda_stream_right_from_file_and_temp_evening.index) > 20:
                eda_evening_mean_right, eda_evening_scrs_right, eda_evening_scrs_mean_ampl_right, eda_evening_scl_cvx_mean_right, eda_evening_scr_cvx_mean_right = calculate_one_hand_eda_features(
                    eda_stream_right_from_file_and_temp_evening)
                df_features.ix[start_time, 'eda_motionless_evening_mean_right'] = eda_evening_mean_right
                df_features.ix[start_time, 'eda_motionless_evening_scrs_right'] = eda_evening_scrs_right
                df_features.ix[start_time, 'eda_motionless_evening_scrs_mean_ampl_right'] = eda_evening_scrs_mean_ampl_right
                df_features.ix[start_time, 'eda_motionless_evening_scl_cvx_mean_right'] = eda_evening_scl_cvx_mean_right
                df_features.ix[start_time, 'eda_motionless_evening_scr_cvx_mean_right'] = eda_evening_scr_cvx_mean_right

            # calculate night EDA features
            if len(eda_stream_right_from_file_and_temp_night.index) > 20:
                eda_night_mean_right, eda_night_scrs_right, eda_night_scrs_mean_ampl_right, eda_night_scl_cvx_mean_right, eda_night_scr_cvx_mean_right = calculate_one_hand_eda_features(
                    eda_stream_right_from_file_and_temp_night)
                df_features.ix[start_time, 'eda_motionless_night_mean_right'] = eda_night_mean_right
                df_features.ix[start_time, 'eda_motionless_night_scrs_right'] = eda_night_scrs_right
                df_features.ix[start_time, 'eda_motionless_night_scrs_mean_ampl_right'] = eda_night_scrs_mean_ampl_right
                df_features.ix[start_time, 'eda_motionless_night_scl_cvx_mean_right'] = eda_night_scl_cvx_mean_right
                df_features.ix[start_time, 'eda_motionless_night_scr_cvx_mean_right'] = eda_night_scr_cvx_mean_right


    if has_enought_recording_from_left_to_process and has_enought_recording_from_right_to_process:
        df_eda_acc_vm_stream_both_combined_motionless = df_eda_acc_vm_stream_left_combined_motionless.join(df_eda_acc_vm_stream_right_combined_motionless, how='inner',
                                                                      lsuffix='_left', rsuffix='_right')
        if len(df_eda_acc_vm_stream_both_combined_motionless.index) > 20:
            eda_24hrs_mean_difference_r_l, eda_24hrs_scrs_diff_r_l, eda_24hrs_scrs_mean_ampl_diff_r_l, eda_24hrs_scl_cvx_mean_diff_r_l, eda_24hrs_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                df_eda_acc_vm_stream_both_combined_motionless)

            recording_time_fraction_24hrs_merged = float(
                len(df_eda_acc_vm_stream_both_combined_motionless.index)) / max_24hrs_signal_length
            df_features.ix[start_time, 'recording_time_fraction_24hrs_merged_motionless'] = recording_time_fraction_24hrs_merged
            df_features.ix[start_time, 'eda_motionless_24hrs_mean_difference_r_l'] = eda_24hrs_mean_difference_r_l
            df_features.ix[start_time, 'eda_motionless_24hrs_scrs_diff_r_l'] = eda_24hrs_scrs_diff_r_l
            df_features.ix[start_time, 'eda_motionless_24hrs_scrs_mean_ampl_diff_r_l'] = eda_24hrs_scrs_mean_ampl_diff_r_l
            df_features.ix[start_time, 'eda_motionless_24hrs_scl_cvx_mean_diff_r_l'] = eda_24hrs_scl_cvx_mean_diff_r_l
            df_features.ix[start_time, 'eda_motionless_24hrs_scr_cvx_mean_diff_r_l'] = eda_24hrs_scr_cvx_mean_diff_r_l


            df_eda_night_merged = df_eda_acc_vm_stream_both_combined_motionless.ix[start_time:start_morning_time,]
            df_eda_morning_merged = df_eda_acc_vm_stream_both_combined_motionless.ix[start_morning_time:start_afternoon_time,]
            df_eda_afternoon_merged = df_eda_acc_vm_stream_both_combined_motionless.ix[start_afternoon_time:start_evening_time,]
            df_eda_evening_merged = df_eda_acc_vm_stream_both_combined_motionless.ix[start_evening_time:end_time,]


            if len(df_eda_morning_merged.index) > 20:
                eda_morning_mean_difference_r_l, eda_morning_scrs_diff_r_l, eda_morning_scrs_mean_ampl_diff_r_l, eda_morning_scl_cvx_mean_diff_r_l, eda_morning_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                    df_eda_morning_merged)

                recording_time_fraction_morning_merged = float(
                    len(df_eda_morning_merged.index)) / max_morning_afternoon_evening_night_signal_length
                df_features.ix[
                    start_time, 'recording_time_fraction_morning_merged_motionless'] = recording_time_fraction_morning_merged
                df_features.ix[start_time, 'eda_motionless_morning_mean_difference_r_l'] = eda_morning_mean_difference_r_l
                df_features.ix[start_time, 'eda_motionless_morning_scrs_diff_r_l'] = eda_morning_scrs_diff_r_l
                df_features.ix[start_time, 'eda_motionless_morning_scrs_mean_ampl_diff_r_l'] = eda_morning_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_motionless_morning_scl_cvx_mean_diff_r_l'] = eda_morning_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_motionless_morning_scr_cvx_mean_diff_r_l'] = eda_morning_scr_cvx_mean_diff_r_l

            if len(df_eda_afternoon_merged.index) > 20:
                eda_afternoon_mean_difference_r_l, eda_afternoon_scrs_diff_r_l, eda_afternoon_scrs_mean_ampl_diff_r_l, eda_afternoon_scl_cvx_mean_diff_r_l, eda_afternoon_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                    df_eda_afternoon_merged)

                recording_time_fraction_afternoon_merged = float(
                    len(df_eda_afternoon_merged.index)) / max_morning_afternoon_evening_night_signal_length
                df_features.ix[
                    start_time, 'recording_time_fraction_afternoon_merged_motionless'] = recording_time_fraction_afternoon_merged
                df_features.ix[start_time, 'eda_motionless_afternoon_mean_difference_r_l'] = eda_afternoon_mean_difference_r_l
                df_features.ix[start_time, 'eda_motionless_afternoon_scrs_diff_r_l'] = eda_afternoon_scrs_diff_r_l
                df_features.ix[
                    start_time, 'eda_motionless_afternoon_scrs_mean_ampl_diff_r_l'] = eda_afternoon_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_motionless_afternoon_scl_cvx_mean_diff_r_l'] = eda_afternoon_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_motionless_afternoon_scr_cvx_mean_diff_r_l'] = eda_afternoon_scr_cvx_mean_diff_r_l

            if len(df_eda_evening_merged.index) > 20:
                eda_evening_mean_difference_r_l, eda_evening_scrs_diff_r_l, eda_evening_scrs_mean_ampl_diff_r_l, eda_evening_scl_cvx_mean_diff_r_l, eda_evening_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                    df_eda_evening_merged)

                recording_time_fraction_evening_merged = float(
                    len(df_eda_evening_merged.index)) / max_morning_afternoon_evening_night_signal_length
                df_features.ix[
                    start_time, 'recording_time_fraction_evening_merged_motionless'] = recording_time_fraction_evening_merged
                df_features.ix[start_time, 'eda_motionless_evening_mean_difference_r_l'] = eda_evening_mean_difference_r_l
                df_features.ix[start_time, 'eda_motionless_evening_scrs_diff_r_l'] = eda_evening_scrs_diff_r_l
                df_features.ix[start_time, 'eda_motionless_evening_scrs_mean_ampl_diff_r_l'] = eda_evening_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_motionless_evening_scl_cvx_mean_diff_r_l'] = eda_evening_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_motionless_evening_scr_cvx_mean_diff_r_l'] = eda_evening_scr_cvx_mean_diff_r_l

            if len(df_eda_night_merged.index) > 20:
                eda_night_mean_difference_r_l, eda_night_scrs_diff_r_l, eda_night_scrs_mean_ampl_diff_r_l, eda_night_scl_cvx_mean_diff_r_l, eda_night_scr_cvx_mean_diff_r_l = calculate_both_hands_eda_features(
                    df_eda_night_merged)

                recording_time_fraction_night_merged = float(
                    len(df_eda_night_merged.index)) / max_morning_afternoon_evening_night_signal_length
                df_features.ix[
                    start_time, 'recording_time_fraction_night_merged_motionless'] = recording_time_fraction_night_merged
                df_features.ix[start_time, 'eda_motionless_night_mean_difference_r_l'] = eda_night_mean_difference_r_l
                df_features.ix[start_time, 'eda_motionless_night_scrs_diff_r_l'] = eda_night_scrs_diff_r_l
                df_features.ix[start_time, 'eda_motionless_night_scrs_mean_ampl_diff_r_l'] = eda_night_scrs_mean_ampl_diff_r_l
                df_features.ix[start_time, 'eda_motionless_night_scl_cvx_mean_diff_r_l'] = eda_night_scl_cvx_mean_diff_r_l
                df_features.ix[start_time, 'eda_motionless_night_scr_cvx_mean_diff_r_l'] = eda_night_scr_cvx_mean_diff_r_l

    return df_features

def generate_empatica_motion_variable_from_y_fast(df, column_name = 'empatica_motion_vector', rolling_window = 8):
    print ("Start generating Empatica motion vector")
    # filter data with a butterworth band pass filter (0.1 Hz - 20Hz band, 4 poles)
    df_acc_filtered = df.apply(apply_butter_filter)

    idx_range_acc_filtered = range(0, len(df_acc_filtered))
    df_acc_filtered_subsampled = df_acc_filtered.iloc[idx_range_acc_filtered[0::subsampling_factor]]

    # df_differ = df.ix[:,0:3]
    df_differ = df_acc_filtered_subsampled.ix[:,'z']
    df_differ = df_differ.diff().abs()
    # df_differ_max = df_differ.max(axis = 1)

    df_differ_max_zsumowane = pd.rolling_sum(df_differ,window=rolling_window)
    df_differ_max_zsumowane.fillna(0, inplace = True)
    differ_max_zsumowane_avg = lfilter([0.1/rolling_window], [1, -0.9], df_differ_max_zsumowane)
    df_differ_max_zsumowane_avg = pd.DataFrame(index = df_differ_max_zsumowane.index, data = differ_max_zsumowane_avg, columns = [column_name])
    return df_differ_max_zsumowane_avg


def calculate_24hrs_eda_features_without_motion():

#   read eda
#   read motion
#   annotate user in motion
#   join motion and eda
# DETECT WHEN USER WAS IMMOBILE AT LEAST 1 MINUTE

    acc_first_row_left = pd.read_hdf(acc_filepath, 'ACC_left', start=0, stop=1)
    start_date_left = acc_first_row_left.index[0]
    acc_last_row_left = pd.read_hdf(acc_filepath, 'ACC_left', start=-2, stop=-1)
    end_date_left = acc_last_row_left.index[0]  # for final version remove that line and uncomment previous line

    acc_first_row_right = pd.read_hdf(acc_filepath, 'ACC_right', start=0, stop=1)
    start_date_right = acc_first_row_right.index[0]
    acc_last_row_right = pd.read_hdf(acc_filepath, 'ACC_right', start=-2, stop=-1)
    end_date_right = acc_last_row_right.index[0]

    start_date = min(start_date_left, start_date_right)
    start_date = start_date.replace(hour=0, minute=0, second=0)
    end_date = max(end_date_left, end_date_right)
    end_date = end_date.replace(hour=0, minute=0, second=0)
    rng = pd.date_range(start_date, end_date)

    df_eda_features = pd.DataFrame()
    for idx, beginning in enumerate(rng):
        # if idx != 23 and idx !=35 and idx !=38 and idx !=49:#M033
        # if idx != 52:#M029
        # if idx != 17 and idx !=21:#M005
        if idx != 41 and idx !=46 and idx !=60 and idx !=77:#M011
        # if idx != 48:#M014
        # if idx != 68:#M020
        # if idx != 52:#M029
        # if idx != 11 and idx!= 40 and idx!= 50 and idx!= 77:#M022
            print (idx, beginning)
            end = beginning + timedelta(hours=24)

            df_24hrs_features =  read_eda_acc_hdfs_and_calculate_features(acc_filepath, eda_filepath, temp_filepath, beginning, end)
            df_eda_features = df_eda_features.append(df_24hrs_features)
    save_features = input('Do you want to save the calculated features (y/n): ') #raw_input in python 2
    if (save_features == 'y'):
        with open(eda_motionless_features_filepath, 'a') as f:
            df_eda_features.to_csv(f, header=True)  # for the blank file change header to True, to write column names


def main():
    print (user_id)
    # calculate_24hrs_eda_features()
    calculate_24hrs_eda_features_without_motion()

if __name__ == "__main__":
    main()