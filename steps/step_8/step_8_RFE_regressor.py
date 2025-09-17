import logging

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from steps.step_generic_code.general_variables.general_variables_all_regressors import REGRESSORS_WITH_PARAMETERS
from steps.step_8.step_8_generic_RFE import do_RFE, store_RFE_results, store_features_ranking,\
                                            get_reduced_frames, get_Y_frames, store_score_result

def strore_score_for_set(app, Y_frames, rfe_result_id, train_or_test):
    Y_set = 'Y_' + train_or_test
    pred = 'pred_' + train_or_test

    mse = mean_squared_error(Y_frames[Y_set],Y_frames[pred])
    store_score_result(app, 'mean_square_error', mse, rfe_result_id, train_or_test)
    r2s = r2_score(Y_frames[Y_set], Y_frames[pred])
    store_score_result(app, 'r2_score', r2s, rfe_result_id, train_or_test)
    mape = mean_absolute_percentage_error(Y_frames[Y_set],Y_frames[pred])
    store_score_result(app, 'mean_absolute_percentage_error', mape, rfe_result_id, train_or_test)

def store_score(app, fit, features, dataframes, rfe_result_id):
    Y_frames = get_Y_frames(fit, features, dataframes['X'], dataframes['Y'], False)
    strore_score_for_set(app, Y_frames, rfe_result_id, 'test')
    strore_score_for_set(app, Y_frames, rfe_result_id, 'train')

def RFE_regressors(app, dataframes, nr_of_classes, fitted_models):
    for name, _, usefitted, repeat, max_result in REGRESSORS_WITH_PARAMETERS:
        model = fitted_models[name]
        print('Step 8: RFE for estimator: ' + name)
        for i in range(repeat):
            logging.info(name)
            reduced_dataframes = dataframes
            for nr_of_features in range(44, 3, -2 ):
                features = reduced_dataframes['X'].columns.tolist()
                fit = do_RFE(reduced_dataframes['X_train'], reduced_dataframes['Y_train'], model, usefitted, nr_of_features)
                rfe_results_id = store_RFE_results(app, fit, name, repeat, nr_of_classes)
                store_features_ranking(app, fit, features, rfe_results_id, last=(nr_of_features==4), is_max=(nr_of_features==max_result))
                store_score(app, fit, features, reduced_dataframes, rfe_results_id)
                reduced_dataframes = get_reduced_frames(fit, features, reduced_dataframes)
