from my_constants import *
from HDRS_prediction.boosting import *
from HDRS_prediction.rf import *

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import ensemble
from my_constants import *
from dimensionality_reduction.dimensionality_reduction import reduce_dimensionality
from sklearn.model_selection import KFold


np.random.seed(SEED)

BEST_VALIDATION_RMSE = 1000
BEST_X = None
BEST_Y = None
BEST_TTL = None
BEST_MDL_NAME = None
BEST_MDL = None


HAMD_file = HAMD_FILENAME






def quick_predict_boosting(inp_x, inp_y, ttl, mdl, ind_train, ind_test):

    global BEST_VALIDATION_RMSE, BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL

    x = np.array(inp_x)[ind_train]
    y = np.array(inp_y)[ind_train]
    x_test = np.array(inp_x)[ind_test]
    y_test = np.array(inp_y)[ind_test]

    # Create linear regression object
    if mdl == 'adaBoost':
        regr = ensemble.GradientBoostingRegressor(random_state=SEED)
    elif mdl=='gb':
        regr = ensemble.AdaBoostRegressor(random_state=SEED)
    elif 'adaBoost' in mdl:
        n = int(mdl[mdl.find('_n')+2:mdl.find('_r')])
        r = float(mdl[mdl.find('_r')+2:mdl.find('_l')])
        l = mdl[mdl.find('_l')+2:]
        regr = ensemble.GradientBoostingRegressor(loss=l, learning_rate=r, n_estimators=n, random_state=SEED)
    elif 'gb' in mdl:
        n = int(mdl[mdl.find('_n')+2:mdl.find('_r')])
        r = float(mdl[mdl.find('_r')+2:mdl.find('_l')])
        l = mdl[mdl.find('_l')+2:]
        regr = ensemble.AdaBoostRegressor(n_estimators=n, learning_rate=r, loss=l, random_state=SEED)

    inds = np.arange(len(y))
    kf = KFold(n_splits=K_FOLD_N)
    splits = kf.split(inds)
    avg_train_RMSE = 0
    avg_validation_RMSE = 0

    for train_inds, validation_inds in splits:
        x_train = x[train_inds]
        y_train = y[train_inds]
        x_validation = x[validation_inds]
        y_validation = y[validation_inds]

        # Train the model using the training sets
        try:
            regr.fit(x_train, y_train)

            validation_RMSE = np.sqrt(mean_squared_error(y_validation, np.round(regr.predict(x_validation))))
            train_RMSE = np.sqrt(mean_squared_error(y_train, np.round(regr.predict(x_train))))

            avg_train_RMSE += train_RMSE
            avg_validation_RMSE += validation_RMSE
        except:
            print ('not converged')

    avg_train_RMSE /= K_FOLD_N
    avg_validation_RMSE /= K_FOLD_N

    regr.fit(x, y)

    if avg_validation_RMSE < BEST_VALIDATION_RMSE:
        BEST_X = inp_x
        BEST_Y = inp_y
        BEST_TTL = ttl
        BEST_MDL_NAME = mdl
        BEST_MDL = regr
        BEST_VALIDATION_RMSE = avg_validation_RMSE

    test_RMSE = np.sqrt(mean_squared_error(y_test, np.round(regr.predict(x_test))))
    print (mdl+', '+ttl+', train RMSE: %f, validation RMSE: %f, test RMSE: %f ' %(avg_train_RMSE, avg_validation_RMSE, test_RMSE))


def quick_plot_prediction_boosting(x, y, ttl, mdl_name, mdl, validation_RMSE, ind_train, ind_test, HAMD_file, nUsers):
    MODEL_FILE_NAME = MODEL_FILE[0:-4] + '_' +HAMD_file[0:-4]+'_learning_curve_boosting.txt'

    test_RMSE = np.sqrt(mean_squared_error(np.array(y)[ind_test], np.round(mdl.predict(np.array(x)[ind_test]))))

    model_file = open(MODEL_FILE_NAME, "a+")
    model_file.write('\nN: '+str(nUsers)+', Best Model: '+mdl_name+', '+ttl+', validation RMSE: %f, test RMSE: %f \n' %(validation_RMSE, test_RMSE))
    model_file.close()


def quick_run_prediction_boosting(HAMD_file, nUsers=len(MDD)):

    all_df = pd.read_csv(data_dir+HAMD_file)
    feature_df = pd.read_csv(feature_dir+'daily_all.csv')
    all_df = all_df.merge(feature_df, on=['ID', 'date'], how='outer')
    all_df = all_df.dropna(subset=[IMPUTATION_LABEL_COMBINATION])
    print(len(all_df.columns))

    all_df = convert_one_hot_str(all_df, 'ID')
    for remUserInd in np.arange(nUsers, len(MDD)):
        all_df = all_df[all_df['ID']!=MDD[remUserInd]]
    all_df = all_df.reset_index(drop=False)


    y_df = all_df[['ID', 'individualized_'+PREDICTION_LABEL, 'original_'+PREDICTION_LABEL, 'baseline_'+PREDICTION_LABEL, 'date', 'imputed',]]
    x_df = all_df.drop(['ID', 'individualized_'+PREDICTION_LABEL, 'original_'+PREDICTION_LABEL, 'baseline_'+PREDICTION_LABEL, 'date', 'imputed'], inplace=False, axis=1)
    x_df_nonan = x_df.fillna(0)

    remove_col = []
    for i in np.arange(len(x_df_nonan.columns.values)):
        if np.std(x_df_nonan[x_df_nonan.columns[i]])==0:
            remove_col.append(i)
    x_df_nonan = x_df_nonan.drop(x_df_nonan.columns[remove_col], axis=1)

    col_num = len(x_df_nonan.columns)
    x_df_nonan, reduced_x_df, reduced_n = reduce_dimensionality(x_df_nonan, max_n=np.min([MAX_PCA_ALL, col_num]), threshold=EXPLAINED_VARIANCE_THRESHOLD)

    y = y_df[[PREDICTION_LABEL_COMBINATION]]
    all_x = x_df_nonan
    pca_x = reduced_x_df[['PCA_'+str(i) for i in np.arange(reduced_n)]]
    kernel_pca_x = reduced_x_df[['KernelPCA_'+str(i) for i in np.arange(reduced_n)]]
    truncated_svd_x = reduced_x_df[['TruncatedSVD_'+str(i) for i in np.arange(reduced_n)]]

    all_columns = x_df_nonan.columns.values
    sub_columns = []
    for col in all_columns:
        if 'sleep' in col: #'daily' in col or
            sub_columns.append(col)
    #sub_x = x_df_nonan[sub_columns]
    # sub_x = x_df_nonan[SUB_FEATURES]

    remove_col = []
    for i in np.arange(len(x_df_nonan.columns.values)):
        if x_df_nonan.columns[i] in SUB_FEATURES:
            continue
        remove_col.append(i)
    sub_x = x_df_nonan.drop(x_df_nonan.columns[remove_col], axis=1)

    sub_col_num = len(sub_x.columns)
    sub_x, reduced_sub_x_df, reduced_sub_n = reduce_dimensionality(sub_x, max_n=np.min([MAX_PCA_SUB, sub_col_num]), threshold=EXPLAINED_VARIANCE_THRESHOLD)
    pca_sub_x = reduced_sub_x_df[['PCA_'+str(i) for i in np.arange(reduced_sub_n)]]
    kernel_pca_sub_x = reduced_sub_x_df[['KernelPCA_'+str(i) for i in np.arange(reduced_sub_n)]]
    truncated_svd_sub_x = reduced_sub_x_df[['TruncatedSVD_'+str(i) for i in np.arange(reduced_sub_n)]]

    sub_x_prev_day = sub_x.shift(periods=1)
    sub_x_prev_day.iloc[0] = sub_x_prev_day.iloc[1]
    if (nUsers<=2):
        ONE_HOT_USERS_SUBSET = [] #it has already been removed due to no variation
    else:
        ONE_HOT_USERS_SUBSET = ['ID_' + user for user in MDD[0:nUsers] if (user != 'M005' and user != 'M031')]
    print (sub_x_prev_day.columns.values)
    sub_x_prev_day.drop(ONE_HOT_USERS_SUBSET, inplace=True, axis=1)
    cols = sub_x_prev_day.columns.values
    print (cols)
    sub_x_prev_day.columns = [col+'_hist' for col in cols]
    sub_x_2 = sub_x.join(sub_x_prev_day)

    sub_col_num_2 = len(sub_x_2.columns)
    sub_x_2, reduced_sub_x_2_df, reduced_sub_n_2 = reduce_dimensionality(sub_x_2, max_n=np.min([MAX_PCA_SUB_HIST, sub_col_num_2]), threshold=EXPLAINED_VARIANCE_THRESHOLD)
    pca_sub_x_2 = reduced_sub_x_2_df[['PCA_'+str(i) for i in np.arange(reduced_sub_n_2)]]
    kernel_pca_sub_x_2 = reduced_sub_x_2_df[['KernelPCA_'+str(i) for i in np.arange(reduced_sub_n_2)]]
    truncated_svd_sub_x_2 = reduced_sub_x_2_df[['TruncatedSVD_'+str(i) for i in np.arange(reduced_sub_n_2)]]


    inds = np.arange(len(y))
    # imputed_inds = y_df[y_df['imputed']=='y'].index
    # for i in imputed_inds:
    #     inds.remove(i)
    # ind_train, ind_test = split_data_ind(inds, int(TEST_RATIO*len(y)))
    # ind_train = list(ind_train) + list(imputed_inds)
    ind_train, ind_test = split_data_ind(inds, int(TEST_RATIO*len(y)))

    print ('\n dataset size:')
    print (len(y))
    print ('\ntrain indices:')
    print (ind_train)
    print ('\ntest indices:')
    print (ind_test)

    models = ['adaBoost', 'gb']
    # adding boosting models
    # n_estimators = [25, 50, 75, 100, 500]
    # learning_rates = [5.0, 1.0, 0.1, 0.001, 0.0001]
    # losses = ['linear', 'square']
    # for n in n_estimators:
    #     for r in learning_rates:
    #         for l in losses:
    #             models.append('adaBoost_n'+str(n)+'_r'+str(r)+'_l'+str(l))
    #     #AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)[source]
    #
    # n_estimators = [50, 100, 200, 500]
    # learning_rates = [5.0, 1.0, 0.1, 0.01, 0.001]
    # losses = ['ls', 'lad', 'huber']
    # for n in n_estimators:
    #     for r in learning_rates:
    #         for l in losses:
    #             models.append('gb_n'+str(n)+'_r'+str(r)+'_l'+str(l))
    # #GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')[source]

    for mdl in models:

        quick_predict_boosting(all_x, y, 'all data', mdl, ind_train, ind_test)
        quick_predict_boosting(pca_x, y, 'PCA', mdl, ind_train, ind_test)
        quick_predict_boosting(kernel_pca_x, y, 'Kernel PCA', mdl, ind_train, ind_test)
        quick_predict_boosting(truncated_svd_x, y, 'Truncated SVD', mdl, ind_train, ind_test)

        quick_predict_boosting(sub_x, y, 'sub data', mdl, ind_train, ind_test)
        quick_predict_boosting(pca_sub_x, y, 'PCA sub', mdl, ind_train, ind_test)
        quick_predict_boosting(kernel_pca_sub_x, y, 'Kernel PCA sub', mdl, ind_train, ind_test)
        quick_predict_boosting(truncated_svd_sub_x, y, 'Truncated SVD sub', mdl, ind_train, ind_test)

        quick_predict_boosting(sub_x_2, y, 'sub hist data', mdl, ind_train, ind_test)
        quick_predict_boosting(pca_sub_x_2, y, 'PCA sub hist', mdl, ind_train, ind_test)
        quick_predict_boosting(kernel_pca_sub_x_2, y, 'Kernel PCA sub hist', mdl, ind_train, ind_test)
        quick_predict_boosting(truncated_svd_sub_x_2, y, 'Truncated SVD sub hist', mdl, ind_train, ind_test)

    quick_plot_prediction_boosting(BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL, BEST_VALIDATION_RMSE, ind_train, ind_test, HAMD_file, nUsers)




MODEL_FILE_NAME = MODEL_FILE[0:-4] + '_' +HAMD_file[0:-4]+'_learning_curve_boosting.txt'
model_file = open(MODEL_FILE_NAME, "w+")
model_file.close()

for i in np.arange(1, len(MDD)):

    BEST_VALIDATION_RMSE = 1000
    BEST_X = None
    BEST_Y = None
    BEST_TTL = None
    BEST_MDL_NAME = None
    BEST_MDL = None
    quick_run_prediction_boosting(HAMD_file, i)
    print(str(i)+' boosting done')
