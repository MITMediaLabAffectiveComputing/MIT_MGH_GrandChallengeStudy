"""
    Author: Asma G
"""

import pandas as pd
import numpy as np
import os
from my_constants import MyConstants
from geopy.distance import great_circle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

#######################        Daily dataset for all users location (smart)      #######################

DIR = MyConstants.combined_log_dir

def compute_clusters_kmeans(df):
    range_n_clusters = np.arange(2, 20)
    best_score = 0
    #best_clusterer = None
    best_cluster_labels = None

    for n_clusters in range_n_clusters:
        print (n_clusters)
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        print ('clustering ...')
        cluster_labels = clusterer.fit_predict(df[['lat', 'long']])
        print ('done')
        silhouette_avg = silhouette_score(df[['lat', 'long']], cluster_labels)
        if (silhouette_avg>best_score):
            best_score = silhouette_avg
            #best_clusterer = clusterer
            best_cluster_labels = cluster_labels
    df['cluster'] = best_cluster_labels
    return df

def compute_clusters(df):
    print (len(df))
    print ('in dbscan')
    db = DBSCAN(eps=0.3, min_samples=10, metric='haversine').fit(df.as_matrix(columns=['lat', 'long']))
    print ('clusterer built and predicted')
    df['cluster'] = db.labels_
    print('labels returend')
    return df

def is_stationary(series):
    prev_point = (series['lat'] - series['lat_diff'], series['long'] - series['long_diff'])
    curr_point = (series['lat'], series['long'])
    # print ('prev: ', prev_point, 'curr: ', curr_point)
    # print ('diff: ', (series['lat_diff'], series['long_diff']))
    series['distance'] = great_circle(prev_point, curr_point).meters
    if series['time_diff'] == 0:
        series['speed'] = 0
    else:
        series['speed'] = float(series['distance'])/ float(series['time_diff'])
    series['stationary'] = series['speed'] < MyConstants.STATIONARY_SPEED
    return series

def is_home(series):
    curr_point = (series['lat'], series['long'])
    home_point = (series['home_lat'], series['home_long'])
    series['distance_from_home'] = great_circle(home_point, curr_point).meters
    series['is_home'] = series['distance_from_home'] < MyConstants.HOME_DISTANCE_THRESHOLD
    return series

def calculate_daily_location_smart():
    #number of clusters
    #entropy
    #normalized entropy
    #home stay
    #circadian movement
    #transition time
    #total distance

    df_all = pd.DataFrame()
    for dname in os.listdir(DIR):
        if dname.startswith('.') or '-' in dname or '_' in dname:
            continue
        # if dname != 'M046':
        #     continue

        print (dname)

        fname = DIR+dname+'/Location_edited.csv'
        df = pd.read_csv(fname, header=None)
        df.columns = ['timestamp', 'lat', 'long', 'alt', 'acc', 'datetime']
        df.sort_values(by=['timestamp'], inplace=True)
        # timestamp is in seconds
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        df = df[df['acc'] < MyConstants.ACC_THRESHOLD]
        df['time_diff'] = -df['timestamp'].diff(periods=-1) #time_diff is in seconds
        df = df[df['time_diff'] != 0]
        df['lat_diff'] = -df['lat'].diff(periods=-1)
        df['long_diff'] = -df['long'].diff(periods=-1)
        df.dropna(subset=['time_diff', 'lat_diff', 'long_diff', 'lat', 'long'], inplace=True)
        df.reset_index(drop=False, inplace=True)

        print ('calculating stationary points...')
        df = df.apply(is_stationary, axis=1)
        print ('done')
        print ('converting index...')
        df = df.apply(MyConstants.date_index2str, axis=1)
        print ('done')

        stationary_df = df[df['stationary'] == True]
        stationary_df['stationary_time_diff'] = -stationary_df['timestamp'].diff(periods=-1)

        # Assigns a cluster number to each stationary point
        # TODO
        # stationary_df = compute_clusters(stationary_df)

        # TODO
        # user_df = pd.DataFrame(columns=['date', 'cluster_num', 'stationary_lat_std', 'stationary_long_std', 'stationary_avg_std', 'entropy', 'normalized_entorpy', 'home_stay', 'transition_time', 'total_distance'])
        user_df = pd.DataFrame(columns=['date', 'weighted_stationary_lat_std', 'weighted_stationary_long_std', 'weighted_stationary_avg_std', 'weighted_home_stay', 'transition_time', 'total_distance'])

        dates = np.unique(df['date'])

        # home location
        mask = stationary_df.datetime.apply(lambda x: x.hour > 0 and x.hour <= 6)
        home_df = stationary_df.loc[mask]
        HOME_LAT = np.median(home_df['lat'])
        HOME_LONG = np.median(home_df['long'])

        for dt in dates:
            print ('day '+dt)
            day_df = df[df['date']==dt]
            day_df.sort_values(by=['timestamp'], inplace=True)
            day_df.reset_index(drop=True, inplace=True)
            # day_df.iloc[0]['time_diff'] = 60 * 60 * (day_df.iloc[0]['datetime'].hour) + 60 * (
            #     day_df.iloc[0]['datetime'].minute) + (day_df.iloc[0]['datetime'].second)
            stationary_day_df = stationary_df[stationary_df['date']==dt]

            # TODO
            # number of clusters
            # cluster_num = len(np.unique(stationary_day_df['cluster']))

            # stationary location standard deviation
            stationary_lat_std = np.std(stationary_day_df['lat'])
            stationary_long_std = np.std(stationary_day_df['long'])
            stationary_avg_std = (stationary_lat_std+stationary_long_std)/2.0

            # for tmp in stationary_day_df['time_diff']:
            #     if (tmp < 0):
            #         print (tmp)

            if (len(stationary_day_df)==0):
                weighted_stationary_lat_std = np.nan
                weighted_stationary_long_std = np.nan
                weighted_stationary_avg_std = np.nan
            else:
                weighted_stationary_lat_std = np.sqrt(
                    np.cov(stationary_day_df['lat'], aweights=stationary_day_df['time_diff']))
                weighted_stationary_long_std = np.sqrt(
                    np.cov(stationary_day_df['long'], aweights=stationary_day_df['time_diff']))
                weighted_stationary_avg_std = (weighted_stationary_lat_std + weighted_stationary_long_std) / 2.0

            # TODO
            # time-based entropy
            # tmp_df = stationary_day_df['time_diff', 'cluster'].groupby(['cluster']).agg(['np.sum'])
            # tmp_total_time = np.sum(tmp_df['time_diff'])
            # tmp_df['p'] = tmp_df['time_diff']/tmp_total_time
            # tmp_df['plogp'] = tmp_df['p']*np.log(tmp_df['p'])
            # entropy = np.sum(tmp_df['plogp'])
            # normalized_entropy = entropy/np.log(cluster_num)

            #home stay
            # mask = stationary_day_df.datetime.apply(lambda x: x.hour>0 and x.hour<=6)
            # home_df = stationary_day_df.loc[mask]
            # day_df['home_lat'] = np.median(home_df['lat'])
            # day_df['home_long'] = np.median(home_df['long'])
            day_df['home_lat'] = HOME_LAT
            day_df['home_long'] = HOME_LONG
            day_df = day_df.apply(is_home, axis=1)
            home_time_df = day_df[day_df['is_home']==True]
            # print ('day df time ', np.sum(day_df['time_diff']), np.sum(day_df['time_diff']) <= (24*60*60.0))
            # print ('home stay time ', sum(home_time_df['time_diff']), np.sum(home_time_df['time_diff']) <= 24 * 60 * 60)
            # weighted_home_stay = float(np.sum(home_time_df['time_diff'])) / (24*60*60.0) * 100.0
            if (len(day_df)==0):
                weighted_home_stay = np.nan
            else:
                weighted_home_stay = float(np.sum(home_time_df['time_diff'])) / float(np.sum(day_df['time_diff'])) * 100.0
            # home_stay = float(len(day_df[day_df['is_home']==True]))/len(day_df)*100.0

            #circadian movement
            #TODO

            #transition time
            # transition_time = float(len(day_df[day_df['stationary']==False]))/len(day_df)*100.0
            stationary_time_df = day_df[day_df['stationary']==True]
            nonstationary_time_df = day_df[day_df['stationary'] == False]
            # print ('nonstatioray time ', np.sum(nonstationary_time_df['time_diff']), np.sum(nonstationary_time_df['time_diff']) <= 24*60*60)
            # transition_time = float(np.sum(nonstationary_time_df['time_diff'] / (24*60*60.0) )) * 100.0
            if (len(day_df) == 0):
                transition_time = np.nan
            else:
                transition_time = float(np.sum(nonstationary_time_df['time_diff'] / np.sum(day_df['time_diff']))) * 100.0

            #total distance
            total_distance = np.sum(day_df['distance'])

            # tmp_row = pd.DataFrame([dt, cluster_num, stationary_lat_std, stationary_long_std, stationary_avg_std, entropy, normalized_entropy, home_stay, transition_time, total_distance], columns=['date', 'cluster_num', 'stationary_lat_std', 'stationary_long_std', 'stationary_avg_std', 'entropy', 'normalized_entorpy', 'home_stay', 'transition_time', 'total_distance'])
            # tmp_row = pd.DataFrame([[dt, stationary_lat_std, stationary_long_std, stationary_avg_std, home_stay, transition_time, total_distance]], columns=['date', 'stationary_lat_std', 'stationary_long_std', 'stationary_avg_std', 'home_stay', 'transition_time', 'total_distance'])
            tmp_row = pd.DataFrame([[dt, weighted_stationary_lat_std, weighted_stationary_long_std,
                                     weighted_stationary_avg_std, weighted_home_stay, transition_time,
                                     total_distance]],
                                   columns=['date', 'weighted_stationary_lat_std', 'weighted_stationary_long_std',
                                            'weighted_stationary_avg_std', 'weighted_home_stay', 'transition_time',
                                            'total_distance'])

            user_df = user_df.append(tmp_row)


        user_df['ID'] = dname
        # print (user_df)
        #user_df.reset_index(level=0, inplace=True)


        df_all = df_all.append(user_df, ignore_index=True)


    return df_all


df_daily = calculate_daily_location_smart()
df_daily = df_daily.sort_values(['ID', 'date']).reset_index(drop=True)

df_daily.to_csv(MyConstants.feature_dir+'daily_location_smart.csv', index=False)
