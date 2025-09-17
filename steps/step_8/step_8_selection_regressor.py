import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from steps.step_generic_code.general_variables.general_variables_all_regressors import FITTING_PARAMETERS, REGRESSORS, NN_REGRESSORS\
                                                                   , ALGORITHM_PARAMETERS
from steps.step_8.step_8_generic_selection import save_pickle_and_metrics
   
def fit_models(X_train, Y_train):
    print("Step 8 RS: Fitting models")
    fitted_models = {}
    for name, estimator, normalized, repeat in REGRESSORS:
        print("Model: ", name)
        parameters=FITTING_PARAMETERS[name]
        log_text="Model {m}".format(m=name)
        logging.info(log_text)
        fitted_models[name]= GridSearchCV(estimator, parameters, cv=2).fit(X_train, Y_train).best_estimator_
    return fitted_models

def fit_nn_models(X_train, Y_train):
    print("Step 8 RS: Fitting models")
    fitted_models = {}
    for name, estimator, normalized, repeat in NN_REGRESSORS:
        print("Model: ", name)
        parameters=FITTING_PARAMETERS[name]
        log_text="Model {m}".format(m=name)
        logging.info(log_text)
        fitted_models[name]= GridSearchCV(estimator, parameters, cv=2).fit(X_train, Y_train).best_estimator_
    return fitted_models

def get_scores_for_prediction(fitted_models, X_set, Y_set):
    print("Step 8 RS: geting scores")
    models_scoring = {}
    for name, fitted_model in fitted_models.items():
        model_score = {}
        Y_pred = fitted_model.predict(X_set)
        model_score['mean_squared_error'] = mean_squared_error(Y_set,Y_pred)
        model_score['r2_score'] = r2_score(Y_set,Y_pred)
        model_score['mean_absolute_percentage_error'] = mean_absolute_percentage_error(Y_set,Y_pred)
        models_scoring[name] = model_score
    return models_scoring

def regressor_selection(app, folder, dataframes):
    fitted_models = fit_models(dataframes['X_train'], dataframes['Y_train'])
    models_scoring = {}
    models_scoring['Train'] = get_scores_for_prediction(fitted_models, dataframes['X_train'], dataframes['Y_train'])
    models_scoring['Test'] = get_scores_for_prediction(fitted_models, dataframes['X_test'], dataframes['Y_test'])
    save_pickle_and_metrics(app, folder, fitted_models, models_scoring, 0, ALGORITHM_PARAMETERS)
    return fitted_models

def nn_regressor_selection(app, folder, dataframes):
    fitted_models = fit_nn_models(dataframes['X_train'], dataframes['Y_train'])
    models_scoring = {}
    models_scoring['Test'] = get_scores_for_prediction(fitted_models, dataframes['X_test'], dataframes['Y_test'])
    models_scoring['Train'] = get_scores_for_prediction(fitted_models, dataframes['X_train'], dataframes['Y_train'])
    save_pickle_and_metrics(app, folder, fitted_models, models_scoring, 0, ALGORITHM_PARAMETERS)
