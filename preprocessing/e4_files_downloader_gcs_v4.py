"""
    File name: e4_files_downloader.py
    Author: Szymon F
    Description: The script provides several functionalities related to the preprocessing of the MGH Grand Chalenge
     data collected with the E4 sensors. The user can chose one of the following options:
     a) download all the E4 streams (the script keeps track of the downloads fo it we will only download the files which were not downloaded yet
        This option will also extract the .csv strems into separate folder for each user
    b) convert the .csv files into h5 format, provide the participant id and the modality which needs to be converted
    c) convert all the .csv files into h5 format for a specific user, provide the user id 
    d) check if there are any index duplicates. Empatica software had a bug which caused an overlap of the E4 stream. This option
     checks if the E4 sensors measurements from the sae user are not overlapping
     e) its purpose is the same as in option d) but the mothod of checking an overlap is different, based on the pattern of the signal and not on
     the time index (as in option d)
        account and organizes them according to the E4 sensor name.
"""

import time
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import csv
import zipfile
import os
import shutil
from os import listdir
import pandas as pd
import numpy as np
from datetime import date
import datetime
from dateutil import tz
from datetime import datetime as dt
import sys
from operator import add
from scipy import signal
import matplotlib.pyplot as plt


#### variables to set
USERNAME="gcs_mgh@mit.edu"
PASSWORD="Depression125"
STUDY_LINK="https://www.empatica.com/connect/studies_detail.php?id=101"

chromedriver = 'P:\Grand Challenge Funding\Empatica\E4\Tools\chromedriver\chromedriver.exe'
filedir="E:\Grand-Chalenge study\Data\Zip sessions from Empatica Connect"
# filedir="P:\Grand Challenge Funding\Data\Zip sessions from Empatica Connect"
boston_timezone = tz.tzlocal()
utc_timezone = tz.gettz('UTC')

#### variables for device tracker
# device_file_path='P:\Grand Challenge Funding\Empatica\Preliminary data\download_and_processing_trackers\Device Tracker_Master - 20161124.csv'
device_file_path='P:\Grand Challenge Funding\Empatica\Preliminary data\download_and_processing_trackers\Device Tracker_Master - 20170520.csv'
download_file_path='P:\Grand Challenge Funding\Empatica\Preliminary data\download_and_processing_trackers\download_tracker.csv'
process_file_path='P:\Grand Challenge Funding\Empatica\Preliminary data\download_and_processing_trackers\process_tracker.csv'
# download_folder = 'P:\Grand Challenge Funding\Empatica\Preliminary data'
# download_folder = 'P:\Grand Challenge Funding\Data\Zip sessions from Empatica Connect'
download_folder = 'E:\Grand-Chalenge study\Data\Zip sessions from Empatica Connect'
download_file_path_h5_streams=download_folder + '\\' + 'h5_streams'
temp_processing_folder = download_folder + '\\' + 'tmp'

#outputfile="/Users/asma/MGH/valid_duration_summary.csv"

if not os.path.isfile(download_file_path):
    csv_file=open(download_file_path, 'w')
    csv_file.write(",input_type,E4_time,E4_device_id,E4_duration\n")
    csv_file.close()

download_df  = pd.read_csv(download_file_path, index_col=0)
### Downloading files

def add_download_file(input_type, E4_device_id,E4_time, E4_duration):
    if check_download_file(input_type, E4_time, E4_device_id, E4_duration):
        print ('Error. Already downloaded: '+input_type+' '+E4_time +' '+ E4_device_id+' '+E4_duration)
        return
    download_df.loc[len(download_df)]=[input_type, E4_time, E4_device_id, E4_duration]
    download_df.to_csv(download_file_path)


def check_download_file(input_type,  E4_device_id, E4_time, E4_duration):
    #print download_df
    #sys.stdout.flush()
    tmp=download_df[download_df['input_type']==input_type]

    if input_type=='E4':
        tmp=tmp[tmp['E4_time']==E4_time]
        tmp=tmp[tmp['E4_device_id']==E4_device_id]
        tmp = tmp[tmp['E4_duration'] == E4_duration]
        if len(tmp)==1:
            return True
        elif len(tmp)==0:
            return False
        else:
            print ('Error. E4 file is downloaded more than once. Device ID:'+ E4_time + ' ' + E4_device_id)
            return True
    else:
        print ('Error. Unknown input type.')
#Device tracker
#NEED TO ADAPT DEVICE TRACKER COLUMNS
df = pd.read_csv(device_file_path)



# df["E4_right_start_date"] = pd.to_datetime(df["E4_right_start_date"])
# df["E4_right_end_date"] = pd.to_datetime(df["E4_right_end_date"])
# df["E4_left_start_date"] = pd.to_datetime(df["E4_left_start_date"])
# df["E4_left_end_date"] = pd.to_datetime(df["E4_left_end_date"])

def date2str(inp):
    if inp<=0:
        return 'Error'
    if inp<10:
        return '0'+str(inp)
    return str(inp)

def find_E4_participant_ID(E4_ID, date):
    #date=dt.strptime(dateStr, "%Y%m%d")
    #return E4_ID
    res_r=df[df['E4_right_ID']==E4_ID]
    #replace NaN in enddate and add 18 hrs to compare to the 6pm and not midnight
    #print res_r
    if len(res_r.index)>0:
        # replace NaN in enddate and add 18 hrs to compare to the 6pm and not midnight
        if (res_r['E4_right_end_date'].isnull().any()):
            end_dates_from_excel_r = pd.to_datetime(res_r['E4_right_end_date'], utc=True).dt.tz_localize('UTC')
            end_dates_from_excel_r_withoutNaN = end_dates_from_excel_r.fillna(dt.utcnow()).dt.tz_localize('US/Eastern')
        else:
            end_dates_from_excel_r_withoutNaN = pd.to_datetime(res_r['E4_right_end_date'], utc=True).dt.tz_localize(
                'US/Eastern')

        # end_dates_from_excel_r = pd.to_datetime(res_r['E4_right_end_date'],utc=True).dt.tz_localize('UTC')
        # #print end_dates_from_excel_r
        # end_dates_from_excel_r_withoutNaN = end_dates_from_excel_r.fillna(dt.utcnow()).dt.tz_localize('US/Eastern')
        end_dates_from_excel_r_withoutNaN = end_dates_from_excel_r_withoutNaN + datetime.timedelta(days = 1)
        #print end_dates_from_excel_r_withoutNaN
        #res_r=res_r[pd.to_datetime(res_r['E4_right_start_date'],utc=True).dt.tz_localize('US/Eastern')<=date]
        corresponding_starts_r=pd.to_datetime(res_r['E4_right_start_date'],utc=True).dt.tz_localize('US/Eastern')<=date
        #res_r=res_r[pd.to_datetime(res_r['E4_right_start_date'],utc=True).dt.tz_localize('US/Eastern')<=date]
        #res_r=res_r[end_dates_from_excel_r_withoutNaN>=date]
        corresponding_ends_r=end_dates_from_excel_r_withoutNaN>=date
        #res_r=res_r[pd.to_datetime(res_r['E4_right_end_date'],utc=True).dt.tz_localize('US/Eastern')>=date]
        res_r = res_r[corresponding_starts_r & corresponding_ends_r]
        res_r = res_r['subject_ID'].unique()

    res_l=df[df['E4_left_ID']==E4_ID]
    if len(res_l.index) > 0:
        #replace NaN in enddate and add 18 hrs to compare to the 6pm and not midnight
        if (res_l['E4_left_end_date'].isnull().any()):
            end_dates_from_excel_l = pd.to_datetime(res_l['E4_left_end_date'],utc=True, errors='coerce').dt.tz_localize('UTC')
            # end_dates_from_excel_l_withoutNaN = end_dates_from_excel_l.fillna(dt.utcnow()).dt.tz_localize('US/Eastern')
            end_dates_from_excel_l_withoutNaN = pd.to_datetime(end_dates_from_excel_l.fillna(dt.utcnow()), utc=True).dt.tz_localize('US/Eastern')
        else:
            end_dates_from_excel_l_withoutNaN = pd.to_datetime(res_l['E4_left_end_date'],utc=True).dt.tz_localize('US/Eastern')
        end_dates_from_excel_l_withoutNaN = end_dates_from_excel_l_withoutNaN + datetime.timedelta(days=1)
        corresponding_starts_l = pd.to_datetime(res_l['E4_left_start_date'], utc=True).dt.tz_localize('US/Eastern') <= date
        corresponding_ends_l = end_dates_from_excel_l_withoutNaN >= date
        res_l = res_l[corresponding_starts_l & corresponding_ends_l]
        res_l = res_l['subject_ID'].unique()
    if len(res_r)==0 and len(res_l)==0:
        print ('Error: No record matching ID')
        return E4_ID,'Err'


    # print 'subjects'
    # print res_r
    # print res_l
    if len(res_r)+len(res_l)>1:
        # error_message =
        print ('Error: More than one record found matching ', E4_ID ,' and ', date)
        return str(E4_ID),'Err'
    if len(res_r)==1:
        return res_r[0],'right'
    elif len(res_l)==1:
        return res_l[0],'left'
    print ('Other Error')
    return E4_ID,'Err'

#PROCESS tracker
if not os.path.isfile(process_file_path):
    csv_file=open(process_file_path, 'w')
    csv_file.write(",original_file_name, process_date\n")
    csv_file.close()

process_df  = pd.read_csv(process_file_path, index_col=0)

def add_process_file(original_file_name):
    if check_process_file(original_file_name):
        print ('Error. Already processed: '+original_file_name)
        return
    process_df.loc[len(process_df)]=[original_file_name, str(dt.now())]
    process_df.to_csv(process_file_path)

def check_process_file(original_file_name):
    tmp=process_df[process_df['original_file_name']==original_file_name].index
    if len(tmp)==1:
        return True
    elif len(tmp)==0:
        return False
    else:
        print ('Error. File processed more than once. '+original_file_name)
        return True


def download_files():
    chromeOptions = webdriver.ChromeOptions()
    prefs = {"download.default_directory" : filedir}
    chromeOptions.add_experimental_option("prefs",prefs)
    browser = webdriver.Chrome(executable_path=chromedriver, chrome_options=chromeOptions)  # Optional argument, if not specified will search path.


    browser.implicitly_wait(10)

    browser.get('https://www.empatica.com/connect/login.php');
    username = browser.find_element_by_name('username')
    password = browser.find_element_by_name('password')
    username.send_keys(USERNAME)
    password.send_keys(PASSWORD)
    login_attempt = browser.find_element_by_name("login-button")
    login_attempt.submit()

    browser.get(STUDY_LINK)
    browser.execute_script("reloadStudySessions(studyId, 0, 999999999999, false);return false;");
    input("Please confirm all the sessions are loaded in the web browser...") #raw_input for python 2
    sessions=browser.find_element_by_id('sessionList').find_elements_by_xpath(".//tr")
    print ("There are "+ str(len(sessions)) + " sessions on the cloud.")

    for idx, session in enumerate(sessions):
        vals=session.find_elements_by_xpath(".//td")
        E4_time = vals[0].text
        E4_duration = vals[1].text
        E4_device_id = vals[2].text
        if not check_download_file('E4', E4_device_id, E4_time, E4_duration):
            # length=vals[1].text
            div = vals[3].find_element_by_xpath(".//div")
            downloadButton = div.find_elements_by_xpath(".//a")[1]
            link = downloadButton.get_attribute("href")
            browser.get(link)
            add_download_file('E4', E4_device_id, E4_time, E4_duration)
            print (idx, 'Downloading: ', E4_device_id, E4_time, E4_duration)
        else:
            print (idx, 'Session already downloaded: ', E4_device_id, E4_time, E4_duration)
    return browser

### CALCULATE DURATION OF VALID DATA
#LOCAL_TIME_OFFSET=datetime.timedelta(0,6*60*60)
#boston_timezone = tz.gettz('America/New_York')
#boston_timezone = tz.tzlocal()
#utc_timezone = tz.gettz('UTC')


def calc_valid_eda(filedir):
    with open(filedir+'/EDA.csv', "rb") as csvfile:
        datareader = csv.reader(csvfile)
        datareader=list(datareader)
        freq=float(datareader[1][0])
        datareader=datareader[2:]
        count = 0
        for row in datareader:
            if row[0]>0.02 and row[0]<30:
                count += 1
        return len(datareader)/freq, count/freq #in seconds

def calc_valid_other(filedir,signal):
    with open(filedir+'/'+signal+'.csv', "rb") as csvfile:
        datareader = csv.reader(csvfile)
        datareader=list(datareader)
        freq=float(datareader[1][0])
        datareader=datareader[2:]
        count = 0
        for row in datareader:
            if row[0]!=0:
                count += 1
        return count/freq #in seconds


def rename_E4_file(temp_folder,signal,rl,participant_ID, output_filedirname, initial_timestamp):
    #date=datetime.date(2000,1,1)
    # with open(temp_folder+'\\' + signal+'.csv', "rb") as csvfile:
    #     datareader = csv.reader(csvfile)
    #     datareader=list(datareader)
    #     freq=float(datareader[1][0])
    #     duration=len(datareader)/freq
    #     start_date = datetime.datetime.utcfromtimestamp(float(datareader[0][0]))
    #     start_date = start_date.replace(tzinfo = utc_timezone)
    #     start_date = start_date.astimezone(boston_timezone)
    #     end_date=start_date+datetime.timedelta(0,duration)
    # print output_filedirname+"GCS_"+participant_ID+"_"+signal+'_'+rl+"_"+initial_timestamp.strftime("%Y%m%d%H%M%S")+".csv"
    # sys.stdout.flush()

    if not os.path.isdir(output_filedirname):
        os.makedirs(output_filedirname)
    os.rename(temp_folder+'\\' + signal+'.csv',output_filedirname+"GCS_"+participant_ID+"_"+signal+'_'+rl+"_"+initial_timestamp.strftime("%Y%m%d%H%M%S")+".csv")


def structure_files():
    files=listdir(filedir)
    print ('There are ',len(files), 'in the folder ', filedir)
    row_list=[]
    for ind, f in enumerate(files):
        if f[-4:] != ".zip":
            continue
        if check_process_file(f):
            continue
        print (str(ind), f)
        filename = f[0:-4]


        filedirname = filedir + '\\' + f
        #unzip files to a temp folder
        with zipfile.ZipFile(filedirname,"r") as z:
            z.extractall(temp_processing_folder)
        #get initial stream timestamp - use EDA as reference
        csv_eda_stream_from_file = pd.read_csv(temp_processing_folder + '\\' + "EDA.csv", header=None, names=['eda'])
        # generate datetim index from the first value
        csv_eda_initial_timestamp = dt.utcfromtimestamp(csv_eda_stream_from_file.ix[0, 'eda'])
        csv_eda_initial_timestamp = csv_eda_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
        csv_eda_initial_timestamp = csv_eda_initial_timestamp.astimezone(tz.gettz('America/New_York'))

        E4_ID = filename[0:6]
        participant_ID, rl = find_E4_participant_ID(E4_ID, csv_eda_initial_timestamp)
        output_user_folder = filedir + '\\' + participant_ID + '\\' + 'E4\\'
        print (filename +' processed to '+ output_user_folder)
        #rename BVP file
        try:
            rename_E4_file(temp_processing_folder, "BVP", rl, participant_ID, output_user_folder, csv_eda_initial_timestamp)
        except:
            print("Unexpected error:", sys.exc_info())
            #print "empty file BVP ", output_user_folder, rl, participant_ID
        #rename ACC file
        try:
            rename_E4_file(temp_processing_folder, "ACC", rl, participant_ID, output_user_folder,
                           csv_eda_initial_timestamp)
        except:
            print("Unexpected error:", sys.exc_info())
        #rename EDA file
        try:
            rename_E4_file(temp_processing_folder, "EDA", rl, participant_ID, output_user_folder,
                           csv_eda_initial_timestamp)
        except:
            print("Unexpected error:", sys.exc_info())
        #rename TEMP file
        try:
            rename_E4_file(temp_processing_folder, "TEMP", rl, participant_ID, output_user_folder,
                           csv_eda_initial_timestamp)
        except:
            print("Unexpected error:", sys.exc_info())
        #rename HR file
        try:
            rename_E4_file(temp_processing_folder, "HR", rl, participant_ID, output_user_folder,
                           csv_eda_initial_timestamp)
        except:
            print("Unexpected error:", sys.exc_info())
        #rename tage file
        try:
            rename_E4_file(temp_processing_folder, "tags", rl, participant_ID, output_user_folder,
                           csv_eda_initial_timestamp)
        except:
            print("Unexpected error:", sys.exc_info())
        #rename IBI
        try:
            rename_E4_file(temp_processing_folder, "IBI", rl, participant_ID, output_user_folder,
                           csv_eda_initial_timestamp)
        except:
            print("Unexpected error:", sys.exc_info())

        #CHECK IF ONLY TEMP FILE IS LEFT AND THEN EMPTY THE FOLDER
        files_in_temp = listdir(temp_processing_folder)
        if len(files_in_temp) > 1:
            print ("There is an unusual problem - some files from the temp have not been moved:")
            print (files_in_temp)



        add_process_file(f)


        # row={"ID":ID, "date": date, "duration": duration, "valid_EDA": valid_EDA, "valid_BVP": valid_BVP, "valid_TEMP":valid_TEMP, "valid_ACC":valid_ACC}
        # row_list.append(row)
        # df = pd.DataFrame(row_list)
        # grouped=df.groupby(["ID", "date"]).agg([np.sum])
        # grouped.to_csv(outputfile)
def delete_files():
    files=listdir(filedir)
    for f in files:
        if f[-4:]!=".zip":
            continue
        filename=f[0:-4]
        filedirname=filedir+'/'+filename
        os.remove(filedirname+'.zip')

def read_E4():
    # browser=download_files()
    # raw_input('Please confirm files are downloaded successfully...')
    # browser.close()
    # raw_input('Please confirm all files downloaded successfully and now will be structured...')
    # find_E4_participant_ID('A00AE4',dt(2016,8,28, tzinfo=tz.gettz('UTC')))#''6/12/2016', "%m/%d/%Y"))
    structure_files()
    # raw_input('Files are structured. Press any key to delete all the zip files...')
    # #delete_files()

def read_and_store_into_hdf(userID,datatype):

    min_acc_session_duration_sec = 60#seconds
    min_eda_session_duration_sec = 60*20#20minutes
    min_hr_session_duration_sec = 60*5#5 minutes
    min_tags_quantity = 1#at least 1 tag
    min_ibi_quantity = 2#at least 1 ibi - have to take into accout the first header row
    min_temp_session_duration_sec = 60*10#10 minutes


    #MAPISAC KOD DO TEMPERATURE oraz BVP


    if not os.path.isdir(download_file_path_h5_streams +'\\' + userID):
        os.makedirs(download_file_path_h5_streams +'\\' + userID)
    csv_file_path = download_folder + '\\' + userID + '\E4\\'
    files = listdir(csv_file_path)
    file_list = []
    for f in files:
        if datatype in f:
            file_list.append(f)
    print ('Please wait..., now creating the h5 file')
    for f in file_list:
        if 'ACC' in datatype:
            output_hdf_file_accel = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_'  + 'acc' + '.h5'
            csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['x', 'y', 'z'])
            frequency = csv_stream_from_file.ix[1,'x']
            duration_sec = len(csv_stream_from_file.index) / frequency
            if duration_sec > min_acc_session_duration_sec:
                csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'x'])
                csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
                index_acc = pd.date_range(start=csv_initial_timestamp, periods=len(csv_stream_from_file.index), freq='31250U')
                csv_stream_from_file.set_index(index_acc, inplace=True)
                # remove first and last 30 seconds
                csv_stream_from_file.drop(csv_stream_from_file.head(frequency * 30).index, inplace=True)
                csv_stream_from_file.drop(csv_stream_from_file.tail(frequency * 30).index, inplace=True)
                csv_stream_from_file.to_hdf(output_hdf_file_accel, key=datatype, mode='a', format='table', append=True)
            else:
                print ('The Accel session is shorter than the minimum required (' + min_acc_session_duration_sec+' seconds). The session was not added to the hdf file:' + csv_file_path + f)

        elif 'IBI' in datatype:
            output_hdf_file_ibi = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'ibi' + '.h5'
            csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['time_elapsed', 'ibi'])
            ibi_quantity = len(csv_stream_from_file.index)
            if ibi_quantity >= min_ibi_quantity:

                csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'time_elapsed'])
                csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[0])
                # add timestamps
                index_ibi = pd.to_timedelta(csv_stream_from_file.ix[:, 'time_elapsed'], unit='s')
                csv_initial_timestamp = pd.Timestamp(csv_initial_timestamp)
                index_ibi = pd.TimedeltaIndex(index_ibi) + pd.DatetimeIndex(
                    [csv_initial_timestamp] * len(index_ibi))
                se_ibi = pd.Series(data=csv_stream_from_file.ix[:, 'ibi'].values, index=index_ibi)
                se_ibi = se_ibi.convert_objects(convert_numeric=True)
                se_ibi.to_hdf(output_hdf_file_ibi, key=datatype, mode='a', format='table', append=True)
            else:
                print ('The IBI session less than the minimum required (' + min_ibi_quantity + ' ibi). The session was not added to the hdf file:' + csv_file_path + f)

        elif 'tags' in datatype:
            output_hdf_file_tags = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'tags' + '.h5'
            if os.stat(csv_file_path + f).st_size > 0:
                csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['tag'])
                tags_quantity = len(csv_stream_from_file.index)
                if tags_quantity > min_tags_quantity:

                    csv_stream_from_file_as_datetime = pd.Series(index=pd.to_datetime(csv_stream_from_file['tag'], unit='s'))
                    # csv_stream_from_file_as_datetime.index =pd.to_datetime(csv_stream_from_file['tag'],unit='s', utc=True)
                    csv_stream_from_file_as_datetime.index = csv_stream_from_file_as_datetime.index.tz_localize(
                        'UTC').tz_convert('America/New_York')
                    csv_stream_from_file_as_datetime.to_hdf(output_hdf_file_tags, key=datatype, mode='a', format='table', append=True)
                    print ('Saved tags: ', str(len(csv_stream_from_file_as_datetime.index)), ' values to ', output_hdf_file_tags)
                else:
                    print ('The tags session less than the minimum required (' + min_tags_quantity + ' tags). The session was not added to the hdf file:' + csv_file_path + f)
        elif 'EDA' in datatype:
            output_hdf_file_eda = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'eda' + '.h5'
            # read sensor stream (EDA or HR or BVP or TEMP)
            csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['eda'])
            frequency = csv_stream_from_file.ix[1, 'eda']
            duration_sec = len(csv_stream_from_file.index) / frequency
            if duration_sec > min_eda_session_duration_sec:

                # generate datetim index from the first value
                csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'eda'])
                csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                # check the sampling rate
                # eda_sampling_rate = eda_stream_from_file.ix[1, 'x'] # usually 4Hz
                # if eda_sampling_rate != 4:
                #    print 'ATTENTION: The eda sampling is not 4Hz, the index is incorrect. Check the source code.'
                # drop timestamp and sampling rate rows to keep only raw data
                csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
                # add index
                index_eda = pd.date_range(start=csv_initial_timestamp, periods=len(csv_stream_from_file.index), freq='250L')
                csv_stream_from_file.set_index(index_eda, inplace=True)
                # save hdf5
                csv_stream_from_file.to_hdf(output_hdf_file_eda, key=datatype, mode='a', format='table', append=True)
            else:
                print ('The EDA session is shorter than the minimum required (' + min_eda_session_duration_sec + ' seconds). The session was not added to the hdf file:' + csv_file_path + f)
        elif 'HR' in datatype:
            output_hdf_file_hr = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' +  'hr' + '.h5'
            # read sensor stream (EDA or HR or BVP or TEMP)
            csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['hr'])
            frequency = csv_stream_from_file.ix[1, 'hr']
            duration_sec = len(csv_stream_from_file.index) / frequency
            if duration_sec > min_hr_session_duration_sec:
                # generate datetim index from the first value
                csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'hr'])
                csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                # check the sampling rate
                # eda_sampling_rate = eda_stream_from_file.ix[1, 'x'] # usually 4Hz
                # if eda_sampling_rate != 4:
                #    print 'ATTENTION: The eda sampling is not 4Hz, the index is incorrect. Check the source code.'
                # drop timestamp and sampling rate rows to keep only raw data
                csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
                # add index
                index_hr = pd.date_range(start=csv_initial_timestamp, periods=len(csv_stream_from_file.index), freq='1S')
                csv_stream_from_file.set_index(index_hr, inplace=True)
                # save hdf5
                csv_stream_from_file.to_hdf(output_hdf_file_hr, key=datatype, mode='a', format='table', append=True)
            else:
                print ('The HR session is shorter than the minimum required (' + min_hr_session_duration_sec + ' seconds). The session was not added to the hdf file:' + csv_file_path + f)
        elif 'TEMP' in datatype:
            output_hdf_file_temp = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + datatype + 'temp' + '.h5'
            # read sensor stream (EDA or HR or BVP or TEMP)
            csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['temp'])
            frequency = csv_stream_from_file.ix[1, 'temp']
            duration_sec = len(csv_stream_from_file.index) / frequency
            if duration_sec > min_temp_session_duration_sec:
                # generate datetim index from the first value
                csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'temp'])
                csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                # check the sampling rate
                # eda_sampling_rate = eda_stream_from_file.ix[1, 'x'] # usually 4Hz
                # if eda_sampling_rate != 4:
                #    print 'ATTENTION: The eda sampling is not 4Hz, the index is incorrect. Check the source code.'
                # drop timestamp and sampling rate rows to keep only raw data
                csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
                # add index
                index_temp = pd.date_range(start=csv_initial_timestamp, periods=len(csv_stream_from_file.index),
                                         freq='250L')
                csv_stream_from_file.set_index(index_temp, inplace=True)
                # save hdf5
                csv_stream_from_file.to_hdf(output_hdf_file_hr, key=datatype, mode='a', format='table', append=True)
            else:
                print ('The Temp session is shorter than the minimum required (' + min_temp_session_duration_sec + ' seconds). The session was not added to the hdf file:' + csv_file_path + f)


    print ('done!')

def read_and_store_into_hdf(userID):

    min_acc_session_duration_sec = 60  # seconds
    min_eda_session_duration_sec = 60 * 20  # 20minutes
    min_hr_session_duration_sec = 60 * 5  # 5 minutes
    min_tags_quantity = 1  # at least 1 tag
    min_ibi_quantity = 2  # at least 1 ibi - have to take into accout the first header row
    min_temp_session_duration_sec = 60 * 10  # 10 minutes

    if not os.path.isdir(download_file_path_h5_streams + '\\' + userID):
        os.makedirs(download_file_path_h5_streams + '\\' + userID)
        # csv_file_path = download_folder + '\\' + userID + '\E4\\'
        csv_file_path = filedir + '\\' + userID + '\E4\\'
        files = listdir(csv_file_path)
        print ('Will process ' + str(len(files)) + ' files.')
        appended_files = 0
        not_appended = 0
        not_processed = 0
        for ind, f in enumerate(files):
            print (str(ind), f)
            if 'ACC' in f:
                output_hdf_file_accel = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'acc' + '.h5'
                csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['x', 'y', 'z'])
                frequency = csv_stream_from_file.ix[1, 'x']
                duration_sec = len(csv_stream_from_file.index) / frequency
                if duration_sec > min_acc_session_duration_sec:
                    csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'x'])
                    csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                    csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                    csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
                    # drop first and last 30 sec of the singal


                    csv_stream_from_file.drop(csv_stream_from_file.head(int(30 * frequency)).index, inplace=True)
                    csv_stream_from_file.drop(csv_stream_from_file.tail(int(30 * frequency)).index, inplace=True)

                    index_acc = pd.date_range(start=csv_initial_timestamp, periods=len(csv_stream_from_file.index),
                                              freq='31250U')
                    csv_stream_from_file.set_index(index_acc, inplace=True)
                    if 'left' in f:
                        side = 'ACC_left'
                    elif 'right' in f:
                        side = 'ACC_right'
                    else:
                        side = 'none'
                        print ('ERROR: file has no side in its name: ' + f)

                    csv_stream_from_file.to_hdf(output_hdf_file_accel, key=side, mode='a', format='table', append=True)
                    appended_files += 1
                else:
                    print ('The Accel session is shorter than the minimum required (' + str(
                        min_acc_session_duration_sec) + ' seconds). The session was not added to the hdf file:' + csv_file_path + f)
                    not_appended += 1
            elif 'IBI' in f:
                output_hdf_file_ibi = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'ibi' + '.h5'
                csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['time_elapsed', 'ibi'])
                ibi_quantity = len(csv_stream_from_file.index)
                if ibi_quantity >= min_ibi_quantity:

                    csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'time_elapsed'])
                    csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                    csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                    csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[0])
                    # add timestamps
                    index_ibi = pd.to_timedelta(csv_stream_from_file.ix[:, 'time_elapsed'], unit='s')
                    csv_initial_timestamp = pd.Timestamp(csv_initial_timestamp)
                    # print index_ibi
                    # print pd.TimedeltaIndex(index_ibi)
                    # print pd.DatetimeIndex([csv_initial_timestamp] * len(index_ibi))
                    index_ibi =  map(add,index_ibi, [csv_initial_timestamp] * len(index_ibi) )
                    # index_ibi = pd.TimedeltaIndex(index_ibi).tolist() + pd.DatetimeIndex([csv_initial_timestamp] * len(index_ibi)).tolist()
                    se_ibi = pd.Series(data=csv_stream_from_file.ix[:, 'ibi'].values, index=index_ibi)
                    se_ibi = se_ibi.convert_objects(convert_numeric=True)
                    if 'left' in f:
                        side = 'IBI_left'
                    elif 'right' in f:
                        side = 'IBI_right'
                    else:
                        side = 'none'
                        print ('ERROR: file has no side in its name: ' + f)
                    se_ibi.to_hdf(output_hdf_file_ibi, key=side, mode='a', format='table', append=True)
                    appended_files += 1
                else:
                    print ('The IBI session less than the minimum required (' + str(
                        min_ibi_quantity) + ' ibi). The session was not added to the hdf file:' + csv_file_path + f)
                    not_appended += 1
            elif 'tags' in f:
                output_hdf_file_tags = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'tags' + '.h5'
                if os.stat(csv_file_path + f).st_size > 0:
                    csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['tag'])
                    tags_quantity = len(csv_stream_from_file.index)
                    if tags_quantity > min_tags_quantity:

                        csv_stream_from_file_as_datetime = pd.Series(
                            index=pd.to_datetime(csv_stream_from_file['tag'], unit='s'),
                            data=[0] * len(csv_stream_from_file['tag']))
                        # csv_stream_from_file_as_datetime.index =pd.to_datetime(csv_stream_from_file['tag'],unit='s', utc=True)
                        csv_stream_from_file_as_datetime.index = csv_stream_from_file_as_datetime.index.tz_localize(
                            'UTC').tz_convert('America/New_York')
                        if 'left' in f:
                            side = 'TAGS_left'
                        elif 'right' in f:
                            side = 'TAGS_right'
                        else:
                            side = 'none'
                            print ('ERROR: file has no side in its name: ' + f)
                        csv_stream_from_file_as_datetime.to_hdf(output_hdf_file_tags, key=side, mode='a',
                                                                format='table', append=True)
                        appended_files += 1
                        print ('Saved tags: ', str(
                            len(csv_stream_from_file_as_datetime.index)), ' values to ', output_hdf_file_tags)
                    else:
                        print ('The tags session less than the minimum required (' + str(
                            min_tags_quantity) + ' tags). The session was not added to the hdf file:' + csv_file_path + f)
                        not_appended += 1
                else:
                    not_appended += 1
            elif 'EDA' in f:
                output_hdf_file_eda = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'eda' + '.h5'
                # read sensor stream (EDA or HR or BVP or TEMP)
                csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['eda'])
                frequency = csv_stream_from_file.ix[1, 'eda']
                duration_sec = len(csv_stream_from_file.index) / frequency
                if duration_sec > min_eda_session_duration_sec:

                    # generate datetim index from the first value
                    csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'eda'])
                    csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                    csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                    # check the sampling rate
                    # eda_sampling_rate = eda_stream_from_file.ix[1, 'x'] # usually 4Hz
                    # if eda_sampling_rate != 4:
                    #    print 'ATTENTION: The eda sampling is not 4Hz, the index is incorrect. Check the source code.'
                    # drop timestamp and sampling rate rows to keep only raw data
                    csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
                    # drop first 19 min and last 60 sec of the signal
                    csv_stream_from_file.drop(csv_stream_from_file.head(int(frequency * 60 * 19)).index, inplace=True)
                    csv_stream_from_file.drop(csv_stream_from_file.tail(int(frequency * 60)).index, inplace=True)

                    # add index
                    index_eda = pd.date_range(start=csv_initial_timestamp, periods=len(csv_stream_from_file.index),
                                              freq='250L')
                    csv_stream_from_file.set_index(index_eda, inplace=True)
                    if 'left' in f:
                        side = 'EDA_left'
                    elif 'right' in f:
                        side = 'EDA_right'
                    else:
                        side = 'none'
                        print ('ERROR: file has no side in its name: ' + f)


                    # save hdf5
                    csv_stream_from_file.to_hdf(output_hdf_file_eda, key=side, mode='a', format='table', append=True)
                    appended_files += 1
                else:
                    print ('The EDA session is shorter than the minimum required (' + str(
                        min_eda_session_duration_sec) + ' seconds). The session was not added to the hdf file:' + csv_file_path + f)
                    not_appended += 1
            elif 'HR' in f:
                output_hdf_file_hr = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'hr' + '.h5'
                # read sensor stream (EDA or HR or BVP or TEMP)
                csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['hr'])
                frequency = csv_stream_from_file.ix[1, 'hr']
                duration_sec = len(csv_stream_from_file.index) / frequency
                if duration_sec > min_hr_session_duration_sec:
                    # generate datetim index from the first value
                    csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'hr'])
                    csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                    csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                    # check the sampling rate
                    # eda_sampling_rate = eda_stream_from_file.ix[1, 'x'] # usually 4Hz
                    # if eda_sampling_rate != 4:
                    #    print 'ATTENTION: The eda sampling is not 4Hz, the index is incorrect. Check the source code.'
                    # drop timestamp and sampling rate rows to keep only raw data
                    csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
                    # add index
                    index_hr = pd.date_range(start=csv_initial_timestamp, periods=len(csv_stream_from_file.index),
                                             freq='1S')
                    csv_stream_from_file.set_index(index_hr, inplace=True)
                    if 'left' in f:
                        side = 'HR_left'
                    elif 'right' in f:
                        side = 'HR_right'
                    else:
                        side = 'none'
                        print ('ERROR: file has no side in its name: ' + f)
                    # save hdf5
                    csv_stream_from_file.to_hdf(output_hdf_file_hr, key=side, mode='a', format='table', append=True)
                    appended_files += 1
                else:
                    print ('The HR session is shorter than the minimum required (' + str(
                        min_hr_session_duration_sec) + ' seconds). The session was not added to the hdf file:' + csv_file_path + f)
                    not_appended += 1
            elif 'TEMP' in f:
                output_hdf_file_temp = download_file_path_h5_streams + '\\' + userID + '\\' + userID + '_' + 'temp' + '.h5'
                # read sensor stream (EDA or HR or BVP or TEMP)
                csv_stream_from_file = pd.read_csv(csv_file_path + f, header=None, names=['temp'])
                frequency = csv_stream_from_file.ix[1, 'temp']
                duration_sec = len(csv_stream_from_file.index) / frequency
                if duration_sec > min_temp_session_duration_sec:
                    # generate datetim index from the first value
                    csv_initial_timestamp = dt.utcfromtimestamp(csv_stream_from_file.ix[0, 'temp'])
                    csv_initial_timestamp = csv_initial_timestamp.replace(tzinfo=tz.gettz('UTC'))
                    csv_initial_timestamp = csv_initial_timestamp.astimezone(tz.gettz('America/New_York'))
                    # check the sampling rate
                    # eda_sampling_rate = eda_stream_from_file.ix[1, 'x'] # usually 4Hz
                    # if eda_sampling_rate != 4:
                    #    print 'ATTENTION: The eda sampling is not 4Hz, the index is incorrect. Check the source code.'
                    # drop timestamp and sampling rate rows to keep only raw data
                    csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
                    # add index
                    index_temp = pd.date_range(start=csv_initial_timestamp, periods=len(csv_stream_from_file.index),
                                               freq='250L')
                    csv_stream_from_file.set_index(index_temp, inplace=True)
                    if 'left' in f:
                        side = 'TEMP_left'
                    elif 'right' in f:
                        side = 'TEMP_right'
                    else:
                        side = 'none'
                        print ('ERROR: file has no side in its name: ' + f)
                    # save hdf5
                    csv_stream_from_file.to_hdf(output_hdf_file_temp, key=side, mode='a', format='table', append=True)
                    appended_files += 1
                else:
                    print ('The Temp session is shorter than the minimum required (' + str(
                        min_temp_session_duration_sec) + ' seconds). The session was not added to the hdf file:' + csv_file_path + f)
                    not_appended += 1
            else:
                print ('ERROR: FILE NOT PROCESSED: ' + f)
                not_processed += 1
        print ('Processing finished: ')
        print ('Files in folder: ' + str(len(files)))
        print ('Appended to hdf: ' + str(appended_files))
        print ('Not appended due to insufficient data: ' + str(not_appended))
        print ('Not processed due to lack of sensor reference: ' + str(not_processed))

def read_hdfs_and_check_duplicate_raws(userID):
    folder_with_hdfs = download_file_path_h5_streams + '\\' + userID
    print ('Checking files in the folder: ',folder_with_hdfs)
    files = listdir(folder_with_hdfs)
    for f in files:
        print (f)
        with pd.HDFStore(folder_with_hdfs + '\\' + f) as hdf:
            stream_keys = hdf.keys()

        for idx, key in enumerate(stream_keys):
            print (key)
            hdf = pd.HDFStore(folder_with_hdfs + '\\' + f)
            nrows = hdf.get_storer(key).nrows
            total_duplicates = 0
            print ('NROWS ', nrows)
            chunksize = 10000000
            total_chunks = nrows // chunksize + 1
            for i in range(total_chunks):               #xrange in python2
                df_chunk = pd.read_hdf(folder_with_hdfs + '\\' + f, key=key, start = i*chunksize, stop = (i + 1) * chunksize)
                # print 'Finished chunk reading.'
                amount_of_duplicates = len(df_chunk[df_chunk.index.duplicated(keep=False)])
                if amount_of_duplicates == 0:
                    print ('There are no duplicates.', float(i)*100/total_chunks, '%')
                else:
                    text =  'There are ',  amount_of_duplicates ,' duplicated indexes:'
                    print (text)
                    total_duplicates+=amount_of_duplicates
                    print (df_chunk[df_chunk.index.duplicated(keep=False)])
            print ('TOTAL DUPLICATES = ',total_duplicates)
            #     print text
            #     print df[df.index.duplicated(keep=False)]
            hdf.close()

def read_csvs_and_check_duplicate_signal_pattern(user_ID):
    comparison_length = 19200  # 10min
    folder_path = filedir + '\\' + user_ID + '\E4\\'

    files = listdir(folder_path)
    print ('Will process ' + str(len(files)) + ' files.')
    last_unprocessed_index = 0
    for ind, f in enumerate(files):
        if ('ACC' in f) and (ind >= last_unprocessed_index ):
        # if ('ACC' in f):
            print (str(ind), f)
            # read acc file
            filepath = folder_path + '\\' + f
            csv_stream_from_file = pd.read_csv(filepath, header=None, names=['x', 'y', 'z'])
            csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
            x_raw = csv_stream_from_file.x.values
            stream_length = len(x_raw)
            if(stream_length>2*comparison_length):
                x_raw_last_ten_pcnt = x_raw[-comparison_length:].copy()
                x_reference = x_raw[0:-comparison_length].copy()
                signal_length = len(x_reference)
                # kod z tej strony: http://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray
                start = time.time()
                windows = rolling_window(x_reference, comparison_length)
                hits = np.ones((len(x_reference) - comparison_length + 1,), dtype=bool)
                # bool_indices = np.all(rolling_window(x_reference, comparison_length) == x_raw_last_ten_pcnt, axis=1)
                # bool_indices = rolling_window(x_reference, comparison_length) == x_raw_last_ten_pcnt.tolist
                for i, x in enumerate(x_raw_last_ten_pcnt):
                    hits &= np.in1d(windows[:, i], [x])
                print ('Processing time (s): ', time.time() - start)
                if (len(hits.nonzero()[0])>0):
                    print ('THIS FILE HAS POTATNIALLY A REPEATED SECTION ########################################################')
                    print (hits.nonzero())
            else:
                print ('File is too short. It has ',stream_length,' samples.')


def main():
    print ('What would you like to do?')
    print ('For download and structure, input a.')
    print ('To convert csv to hdf, input b.')
    print ('To convert all csv to hdf for specific user, input c.')
    print ('To check duplicated raws based on the signal indexes, input d. ')
    print ('To check tail signal (of 30min) repetition, input e. ')
    key = input()                                   #raw_input in python 2
    if key == 'a':
        return read_E4()
    elif key == 'b':
        user = input('Please enter user ID')        #raw_input in python 2
        datatype = input(
            'Please enter the datatype - EDA_left, TEMP_right, ACC_left, tags_right, IBI_left, HR_right')       #raw_input in python 2
        return read_and_store_into_hdf(user, datatype)
    elif key == 'c':
        user = input('Please enter user ID: ')      #raw_input in python 2
        return read_and_store_into_hdf(user)
    elif key == 'd':
        user = input('Please enter user ID: ')      #raw_input in python 2
        return read_hdfs_and_check_duplicate_raws(user)
    elif key == 'e':
        user = input('Please enter user ID: ')      #raw_input in python 2
        # user = 'M016'
        return read_csvs_and_check_duplicate_signal_pattern(user)
    else:
        print('Please enter a valid option')
        main()


def rolling_window(a, size):
     shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
     strides = a.strides + (a. strides[-1],)
     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def test():
    # a = np.array([0,1,2,3,4,5,6,7,8,9])
    # b = a.argsort()[-3:][::-1]
    # print b
    #
    # ind = np.argpartition(a, -2)[-2:]
    # c =  ind[np.argsort(a[ind])]
    # print c


    filepath = 'C:\Users\Szymon\Downloads\E4 duplicates\A009C6_160629-082246_z\ACC.csv'
    print (filepath)
   # read acc file
    csv_stream_from_file = pd.read_csv(filepath, header=None, names=['x', 'y', 'z'])
    csv_stream_from_file = csv_stream_from_file.drop(csv_stream_from_file.index[:2])
    x_raw = csv_stream_from_file.x.values



    comparison_length =19200#10min

    x_raw_last_ten_pcnt = x_raw[-comparison_length:].copy()
    x_reference = x_raw[0:-comparison_length].copy()
    signal_length = len(x_reference)
    #kod z tej strony: http://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray
    start = time.time()
    windows = rolling_window(x_reference, comparison_length)
    hits = np.ones((len(x_reference) - comparison_length + 1,), dtype=bool)
    # bool_indices = np.all(rolling_window(x_reference, comparison_length) == x_raw_last_ten_pcnt, axis=1)
    # bool_indices = rolling_window(x_reference, comparison_length) == x_raw_last_ten_pcnt.tolist
    for i, x in enumerate(x_raw_last_ten_pcnt):
        if i % 1000 == 0:
            print (i)
        hits &= np.in1d(windows[:, i], [x])
    print (time.time() - start)
    print (len(hits.nonzero()))

    # print np.mgrid[0:len(bool_indices)][bool_indices]

    # index = 0
    # diff_arr = np.array([])
    # print signal_length, len(x_raw_last_ten_pcnt)
    #
    # while(signal_length-index >= comparison_length):
    #     diff_arr = np.append(diff_arr, np.abs(x_reference[index:index + comparison_length] - x_raw_last_ten_pcnt).sum())
    #     if(index % 10000 == 0):
    #         print index, diff_arr
    #
    #     # if(np.abs(x_reference[index:index+comparison_length] - x_raw_last_ten_pcnt).sum()< 100):
    #     #     print 'THIS IS POTENTIALLY CORRUPTED SIGNAL.'
    #     index+=1
    # plot the results
    # fig, (ax_orig, ax_reference, ax_corr) = plt.subplots(3, 1, sharex=True)
    # fig, (ax_orig, ax_reference, ax_corr) = plt.subplots(3, 1)
    # ax_orig.plot(x_reference, color='red')
    # ax_orig.set_ylim([-100,100])
    # ax_reference.plot(x_raw_last_ten_pcnt)
    # ax_reference.set_ylim([-100,100]) # ax_orig.plot(y_raw, color='green')
    # # ax_orig.plot(z_raw, color='blue')
    #
    # ax_corr.plot(diff_arr, color = 'blue', lw=2)
    # # tytul = 'A00974_161213-023249 '
    # # ax_orig.set_title(tytul)
    # plt.show()


    # a = np.array([0, 1, 2, 3, 4, 5, 6])
    # print a[0:-1]
    # print np.abs(a).sum()
    # # b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # # print max(a-b)==min(a-b)==0



if __name__ == "__main__":
    main()
    #test()