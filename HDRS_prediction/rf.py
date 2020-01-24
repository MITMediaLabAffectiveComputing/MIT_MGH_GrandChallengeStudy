"""
    Author: Asma G
"""

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import ensemble
import numpy as np
import pandas as pd
from my_constants import MyConstants
from dimensionality_reduction.dimensionality_reduction import reduce_dimensionality
from sklearn.model_selection import KFold

np.random.seed(MyConstants.get_seed(MyConstants))

BEST_VALIDATION_RMSE = 1000
BEST_X = None
BEST_Y = None
BEST_TTL = None
BEST_MDL_NAME = None
BEST_MDL = None


def predict_rf(inp_x, inp_y, ttl, mdl, ind_train, ind_test, model_file):
    global BEST_VALIDATION_RMSE, BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL

    # standardize input
    inp_x = MyConstants.standardize_df(inp_x)

    x = np.array(inp_x)[ind_train]
    y = np.array(inp_y)[ind_train]
    x_test = np.array(inp_x)[ind_test]
    y_test = np.array(inp_y)[ind_test]

    # Create regression object
    if mdl == 'regression':
        regr = linear_model.LinearRegression()
    elif mdl == 'ridge':
        regr = linear_model.RidgeCV(cv=MyConstants.K_FOLD_N, alphas=MyConstants.REGULARIZATION_ALPHAS)
    elif mdl == 'lasso':
        regr = linear_model.LassoCV(cv=MyConstants.K_FOLD_N, alphas=MyConstants.REGULARIZATION_ALPHAS)
    elif mdl == 'elasticNet':
        regr = linear_model.ElasticNetCV(cv=MyConstants.K_FOLD_N, alphas=MyConstants.REGULARIZATION_ALPHAS)
    elif mdl == 'theil':
        regr = linear_model.TheilSenRegressor(random_state=MyConstants.get_seed(MyConstants))
    elif 'ransac' in mdl:  # ransac_{ms}
        ms = float(mdl[mdl.find('_') + 1:])
        regr = linear_model.RANSACRegressor(random_state=MyConstants.get_seed(MyConstants), min_samples=ms)
    elif 'huber' in mdl:  # huber_e{eps}_a{al}
        eps = float(mdl[mdl.find('_e') + 2:mdl.find('_a')])
        al = float(mdl[mdl.find('_a') + 2:])
        regr = linear_model.HuberRegressor(epsilon=eps, alpha=al)
    elif 'rf' in mdl:  # rf_{n}
        n = int(mdl[mdl.find('_') + 1:])
        regr = ensemble.RandomForestRegressor(random_state=MyConstants.get_seed(MyConstants), n_estimators=n, n_jobs=-1)
    # elif mdl == 'gp':
    #     regr = gaussian_process.GaussianProcessRegressor(n_restarts_optimizer=N_RESTARTS_OPTIMIZER, random_state=SEED)

    inds = np.arange(len(y))
    kf = KFold(n_splits=MyConstants.K_FOLD_N, random_state=MyConstants.get_seed(MyConstants))
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
            print('not converged')

    avg_train_RMSE /= MyConstants.K_FOLD_N
    avg_validation_RMSE /= MyConstants.K_FOLD_N

    regr.fit(x, y)

    if avg_validation_RMSE < BEST_VALIDATION_RMSE:
        BEST_X = inp_x
        BEST_Y = inp_y
        BEST_TTL = ttl
        BEST_MDL_NAME = mdl
        BEST_MDL = regr
        BEST_VALIDATION_RMSE = avg_validation_RMSE

    test_RMSE = np.sqrt(mean_squared_error(y_test, np.round(regr.predict(x_test))))
    print(mdl + ', ' + ttl + ', train RMSE: %f, validation RMSE: %f, test RMSE: %f ' % (
    avg_train_RMSE, avg_validation_RMSE, test_RMSE))
    model_file.write(mdl + ', ' + ttl + ', train RMSE: %f, validation RMSE: %f, test RMSE: %f  \n' % (
    avg_train_RMSE, avg_validation_RMSE, test_RMSE))


def plot_prediction_rf(x, y, ttl, mdl_name, mdl, validation_RMSE, ind_train, ind_test, HAMD_file, modality):
    MODEL_FILE_NAME = MyConstants.MODEL_FILE[0:-4] + '_' + HAMD_file[0:-4] + '_rf.txt'
    if (modality):
        MODEL_FILE_NAME = MyConstants.MODEL_FILE[0:-4] + '_' + HAMD_file[0:-4] + '_rf_' + modality[0] + '.txt'

    test_RMSE = np.sqrt(mean_squared_error(np.array(y)[ind_test], np.round(mdl.predict(np.array(x)[ind_test]))))

    plt.figure(figsize=(16, 4))
    plt.scatter(np.arange(len(y)), y, label=MyConstants.PREDICTION_LABEL, color='green', alpha=0.5)
    plt.scatter(ind_train, np.round(mdl.predict(np.array(x)[ind_train])), label='predicted - train', color='red',
                alpha=0.5)
    plt.scatter(ind_test, np.round(mdl.predict(np.array(x)[ind_test])), label='predicted - test', color='black',
                alpha=0.5)
    plt.title('imputation: ' + HAMD_file[0:-4] + ', model: ' + mdl_name + ', dataset: ' + ttl +
              ', validation RMSE: ' + '{:.3f}'.format(validation_RMSE) +
              ', test RMSE: ' + '{:.3f}'.format(test_RMSE))
    plt.ylabel(MyConstants.PREDICTION_LABEL)
    plt.legend(loc=2, scatterpoints=1)

    model_file = open(MODEL_FILE_NAME, "a+")
    model_file.write('\nBest Model: ' + mdl_name + ', ' + ttl + ', validation RMSE: %f, test RMSE: %f \n' % (
    validation_RMSE, test_RMSE))
    if 'gp' in mdl_name or 'ransac' in mdl_name or 'rf' in mdl_name:
        print('no coefficiants to print for this model.')
    else:
        print('model parameters: \n')
        print(mdl.coef_)
        model_file.write('coefficients:\n')
        for item in mdl.coef_:
            model_file.write('%s \t' % item)
        model_file.write('\n')
    model_file.close()
    fig_title = HAMD_file[0:-4] + '_' + mdl_name + '_' + ttl + \
                '_v_' + '{:.3f}'.format(validation_RMSE) + \
                '_t_' + '{:.3f}'.format(test_RMSE) + '.pdf'
    if (modality):
        fig_title = HAMD_file[0:-4] + '_' + mdl_name + '_' + ttl + \
                    '_v_' + '{:.3f}'.format(validation_RMSE) + \
                    '_t_' + '{:.3f}'.format(test_RMSE) + '_' + modality[0] + '.pdf'

    plt.savefig(MyConstants.prediction_fig_dir + fig_title, transparent=True, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(16, 4))
    plt.scatter(np.arange(len(ind_test)), np.array(y)[ind_test], label=MyConstants.PREDICTION_LABEL, color='green',
                alpha=0.5)
    plt.scatter(np.arange(len(ind_test)), np.round(mdl.predict(np.array(x)[ind_test])), label='predicted - test',
                color='black', alpha=0.5)
    plt.title('imputation: ' + HAMD_file[0:-4] + ', model: ' + mdl_name + ', dataset: ' + ttl +
              ', validation RMSE: ' + '{:.3f}'.format(validation_RMSE) +
              ', test RMSE: ' + '{:.3f}'.format(test_RMSE))
    plt.ylabel(MyConstants.PREDICTION_LABEL)
    plt.legend(loc=2, scatterpoints=1)
    plt.savefig(MyConstants.prediction_fig_dir + 'test/' + fig_title, transparent=True, format='pdf',
                bbox_inches='tight')

    output_df = pd.DataFrame(data=np.round(mdl.predict(np.array(x))), columns=[mdl_name + '_' + ttl])
    output_df['train'] = '_'
    output_df['train'][ind_train] = 'y'
    output_df['train'][ind_test] = 'n'
    if (modality):
        output_df.to_csv(MyConstants.results_dir + 'rf_' + modality[0] + '.csv')
    else:
        output_df.to_csv(MyConstants.results_dir + 'rf.csv')


def run_prediction_rf(HAMD_file, modality=None):
    MODEL_FILE_NAME = MyConstants.MODEL_FILE[0:-4] + '_' + HAMD_file[0:-4] + '_rf.txt'
    if (modality):
        MODEL_FILE_NAME = MyConstants.MODEL_FILE[0:-4] + '_' + HAMD_file[0:-4] + '_rf_' + modality[0] + '.txt'
    all_df = pd.read_csv(MyConstants.data_dir + HAMD_file)
    feature_df = pd.read_csv(MyConstants.feature_dir + 'daily_all.csv')
    all_df = all_df.merge(feature_df, on=['ID', 'date'], how='outer')
    all_df = all_df.dropna(subset=[MyConstants.PREDICTION_LABEL_COMBINATION])
    if (modality):
        remove_col = []
        for i in np.arange(len(all_df.columns.values)):
            if all_df.columns[i] in ['ID', 'group', 'date', 'imputed'] + ['individualized_' + col for col in
                                                                          MyConstants.LABEL_FACTORS] + [
                'original_' + col for col in MyConstants.LABEL_FACTORS] + ['baseline_' + col for col in
                                                                            MyConstants.LABEL_FACTORS]:
                continue
            rm = True
            for m in modality:
                if (m in all_df.columns[i]):
                    rm = False
            if (rm):
                remove_col.append(i)
        all_df = all_df.drop(all_df.columns[remove_col], axis=1)

    all_df = MyConstants.convert_one_hot_str(all_df, 'ID')

    y_df = all_df[['ID', 'individualized_' + MyConstants.PREDICTION_LABEL, 'original_' + MyConstants.PREDICTION_LABEL,
                   'baseline_' + MyConstants.PREDICTION_LABEL, 'date', 'imputed']]
    x_df = all_df.drop(
        ['ID', 'individualized_' + MyConstants.PREDICTION_LABEL, 'original_' + MyConstants.PREDICTION_LABEL,
         'baseline_' + MyConstants.PREDICTION_LABEL, 'date', 'imputed'], inplace=False, axis=1)
    x_df_nonan = x_df.fillna(0)

    remove_col = []
    for i in np.arange(len(x_df_nonan.columns.values)):
        print(x_df_nonan.columns.values[i])
        if np.std(x_df_nonan[x_df_nonan.columns[i]]) == 0:
            remove_col.append(i)
    x_df_nonan = x_df_nonan.drop(x_df_nonan.columns[remove_col], axis=1)

    col_num = len(x_df_nonan.columns)
    x_df_nonan, reduced_x_df, reduced_n = reduce_dimensionality(x_df_nonan,
                                                                max_n=np.min([MyConstants.MAX_PCA_ALL, col_num]),
                                                                threshold=MyConstants.EXPLAINED_VARIANCE_THRESHOLD)

    y = y_df[[MyConstants.PREDICTION_LABEL_COMBINATION]]
    all_x = x_df_nonan
    pca_x = reduced_x_df[['PCA_' + str(i) for i in np.arange(reduced_n)]]
    kernel_pca_x = reduced_x_df[['KernelPCA_' + str(i) for i in np.arange(reduced_n)]]
    truncated_svd_x = reduced_x_df[['TruncatedSVD_' + str(i) for i in np.arange(reduced_n)]]

    all_columns = x_df_nonan.columns.values
    sub_columns = []
    for col in all_columns:
        if 'ID_' in col or 'daily' in col or '24hrs' in col or 'sleep_reg_index' in col or 'weather_' in col:
            sub_columns.append(col)
    sub_x = x_df_nonan[sub_columns]
    # sub_x = x_df_nonan[SUB_FEATURES]

    # remove_col = []
    # for i in np.arange(len(x_df_nonan.columns.values)):
    #     if x_df_nonan.columns[i] in MyConstants.SUB_FEATURES:
    #         continue
    #     remove_col.append(i)
    # sub_x = x_df_nonan.drop(x_df_nonan.columns[remove_col], axis=1)

    sub_col_num = len(sub_x.columns)
    sub_x, reduced_sub_x_df, reduced_sub_n = reduce_dimensionality(sub_x, max_n=np.min(
        [MyConstants.MAX_PCA_SUB, sub_col_num]), threshold=MyConstants.EXPLAINED_VARIANCE_THRESHOLD)
    pca_sub_x = reduced_sub_x_df[['PCA_' + str(i) for i in np.arange(reduced_sub_n)]]
    kernel_pca_sub_x = reduced_sub_x_df[['KernelPCA_' + str(i) for i in np.arange(reduced_sub_n)]]
    truncated_svd_sub_x = reduced_sub_x_df[['TruncatedSVD_' + str(i) for i in np.arange(reduced_sub_n)]]

    sub_x_prev_day = sub_x.shift(periods=1)
    sub_x_prev_day.iloc[0] = sub_x_prev_day.iloc[1]
    sub_x_prev_day.drop(MyConstants.ONE_HOT_USERS, inplace=True, axis=1)
    cols = sub_x_prev_day.columns.values
    sub_x_prev_day.columns = [col + '_hist' for col in cols]
    sub_x_2 = sub_x.join(sub_x_prev_day)

    sub_col_num_2 = len(sub_x_2.columns)
    sub_x_2, reduced_sub_x_2_df, reduced_sub_n_2 = reduce_dimensionality(sub_x_2, max_n=np.min(
        [MyConstants.MAX_PCA_SUB_HIST, sub_col_num_2]), threshold=MyConstants.EXPLAINED_VARIANCE_THRESHOLD)
    pca_sub_x_2 = reduced_sub_x_2_df[['PCA_' + str(i) for i in np.arange(reduced_sub_n_2)]]
    kernel_pca_sub_x_2 = reduced_sub_x_2_df[['KernelPCA_' + str(i) for i in np.arange(reduced_sub_n_2)]]
    truncated_svd_sub_x_2 = reduced_sub_x_2_df[['TruncatedSVD_' + str(i) for i in np.arange(reduced_sub_n_2)]]

    inds = np.arange(len(y))
    # imputed_inds = y_df[y_df['imputed']=='y'].index
    # for i in imputed_inds:
    #     inds.remove(i)
    # ind_train, ind_test = split_data_ind(inds, int(TEST_RATIO*len(y)))
    # ind_train = list(ind_train) + list(imputed_inds)
    ind_train, ind_test = MyConstants.split_data_ind(MyConstants, inds, int(MyConstants.TEST_RATIO * len(y)))

    # print('\n dataset size:')
    # print(len(y))
    # print('\ntrain indices:')
    # print(ind_train)
    # print('\ntest indices:')
    # print(ind_test)

    models = []

    # adding rf models
    n_estimators = [5, 10, 15, 20, 25]
    for n in n_estimators:
        models.append('rf_' + str(n))

    model_file = open(MODEL_FILE_NAME, "w")
    model_file.close()
    for mdl in models:
        model_file = open(MODEL_FILE_NAME, "a+")

        predict_rf(all_x, y, 'all data', mdl, ind_train, ind_test, model_file)
        predict_rf(pca_x, y, 'PCA', mdl, ind_train, ind_test, model_file)
        predict_rf(kernel_pca_x, y, 'Kernel PCA', mdl, ind_train, ind_test, model_file)
        predict_rf(truncated_svd_x, y, 'Truncated SVD', mdl, ind_train, ind_test, model_file)

        predict_rf(sub_x, y, 'sub data', mdl, ind_train, ind_test, model_file)
        predict_rf(pca_sub_x, y, 'PCA sub', mdl, ind_train, ind_test, model_file)
        predict_rf(kernel_pca_sub_x, y, 'Kernel PCA sub', mdl, ind_train, ind_test, model_file)
        predict_rf(truncated_svd_sub_x, y, 'Truncated SVD sub', mdl, ind_train, ind_test, model_file)

        predict_rf(sub_x_2, y, 'sub hist data', mdl, ind_train, ind_test, model_file)
        predict_rf(pca_sub_x_2, y, 'PCA sub hist', mdl, ind_train, ind_test, model_file)
        predict_rf(kernel_pca_sub_x_2, y, 'Kernel PCA sub hist', mdl, ind_train, ind_test, model_file)
        predict_rf(truncated_svd_sub_x_2, y, 'Truncated SVD sub hist', mdl, ind_train, ind_test, model_file)

        model_file.close()

    plot_prediction_rf(BEST_X, BEST_Y, BEST_TTL, BEST_MDL_NAME, BEST_MDL, BEST_VALIDATION_RMSE, ind_train, ind_test,
                       HAMD_file, modality)


# UNCOMMENT IF YOU WANT TO RUN ALL


# HAMD_files = [HAMD_FILENAME]
#
# for HAMD_file in HAMD_files:
#     BEST_VALIDATION_RMSE = 1000
#     BEST_X = None
#     BEST_Y = None
#     BEST_TTL = None
#     BEST_MDL_NAME = None
#     BEST_MDL = None
#     run_prediction_rf(HAMD_file)

def main():
    run_prediction_rf(MyConstants.HAMD_FILENAME)


if __name__ == '__main__':
    main()

# TODO: encode missing data?

# TODO hieriarchichal bayes with STAN

# TODO: sensor high dimentional features -> NN -> new features
