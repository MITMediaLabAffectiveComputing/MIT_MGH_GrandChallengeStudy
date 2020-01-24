"""
    Author: Asma G
"""

import sys
sys.path.insert(0, '.')

from HDRS_prediction.boosting import *
from HDRS_prediction.rf import *
from HDRS_prediction.ensemble_avg import *
from my_constants import  MyConstants




# BEST_VALIDATION_RMSE = 1000
# BEST_X = None
# BEST_Y = None
# BEST_TTL = None
# BEST_MDL_NAME = None
# BEST_MDL = None
# run_prediction_boosting(HAMD_file)
# print (' boosting done')
#
# BEST_VALIDATION_RMSE = 1000
# BEST_X = None
# BEST_Y = None
# BEST_TTL = None
# BEST_MDL_NAME = None
# BEST_MDL = None
# run_prediction_rf(HAMD_file)
# print(' rf done')
#
# run_prediction_avg(HAMD_file)
# print(' avg done')

def main():
    np.random.seed(MyConstants.get_seed(MyConstants))
    HAMD_file = MyConstants.HAMD_FILENAME
    global BEST_VALIDATION_RMSE, BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL

    modality = MyConstants.get_modality(MyConstants)
    print(modality)
    BEST_VALIDATION_RMSE = 1000
    BEST_X = None
    BEST_Y = None
    BEST_TTL = None
    BEST_MDL_NAME = None
    BEST_MDL = None
    run_prediction_boosting(HAMD_file, modality)
    print(modality, ' boosting done')

    BEST_VALIDATION_RMSE = 1000
    BEST_X = None
    BEST_Y = None
    BEST_TTL = None
    BEST_MDL_NAME = None
    BEST_MDL = None
    run_prediction_rf(HAMD_file, modality)
    print(modality, ' rf done')

    run_prediction_avg(HAMD_file, modality)
    print(modality, ' avg done')

    print(modality)

if __name__ == '__main__':
    run_main = True
    replace = 'Y'
    for arg in sys.argv[1:]:
        if 'SEED=' in arg:
            MyConstants.set_seed(MyConstants, int(arg[arg.find('SEED=')+len('SEED='):]))
            print('SEED was set: ' + str(MyConstants.get_seed(MyConstants)))
        if 'TRAIN_TEST_OPTION=' in arg:
            MyConstants.set_train_test_option(MyConstants, int(arg[arg.find('TRAIN_TEST_OPTION=')+len('TRAIN_TEST_OPTION='):]))
            print('TRAIN_TEST_OPTION was set: ' + str(MyConstants.get_train_test_option(MyConstants)))
        if 'MODALITY=' in arg:
            MyConstants.set_modality(MyConstants, arg[arg.find('MODALITY=')+len('MODALITY='):])
            print ('MODALITY was set: ' + str(MyConstants.get_modality(MyConstants)))
        if 'REPLACE=' in arg:
            replace = arg[arg.find('REPLACE=')+len('REPLACE='):]
        if 'HAMD_FILE=' in arg:
            MyConstants.set_HAMD_filename(MyConstants, arg[arg.find('HAMD_FILE=')+len('HAMD_FILE='):])
            print('HAMD_FILE was set: ' + str(MyConstants.HAMD_FILENAME))

    MyConstants._update_directories(MyConstants)
    if replace == 'N':
        results_file = MyConstants.results_dir + 'ensemble_avg_' + MyConstants.get_modality(MyConstants)[0] + '.csv'
        if os.path.exists(results_file):
            print('File already exists: ' + results_file + ' Skipping...')
            run_main = False

    if run_main:
        main()


