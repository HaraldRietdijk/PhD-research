from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd

from steps.step_10.step_10_filter_methods import append_scores, init_scores
from steps.step_generic_code.general_variables.general_variables_all_shap import CLASSIFIERS, FITTING_PARAMETERS

def get_lasso_fitting(dataframes):
    # params = {"alpha":[0.00001, 1, 10, 500]}
    params = {"alpha":[0.0001]}
    lasso_fitting = GridSearchCV(Lasso(), param_grid=params, cv=5).fit(dataframes['X'],dataframes['Y_class']).best_estimator_
    importance = pd.DataFrame([abs(coefficient) for coefficient in lasso_fitting.coef_] ,
                          columns=['coefficient'])
    importance['normalized'] = importance/importance.max()
    return lasso_fitting, importance

def get_coefficient_threshold(importance, threshold):
    valid_coefficients = importance.loc[importance['normalized']>=threshold]
    return valid_coefficients['coefficient'].min()

def get_lasso_features(dataframes, features, lasso_fitting, importance, threshold):
    coefficient_threshold = get_coefficient_threshold(importance, threshold)
    select_from_model = SelectFromModel(lasso_fitting, threshold=coefficient_threshold)
    lasso_features = select_from_model.fit(dataframes['X'],dataframes['Y_class']).get_feature_names_out(features)
    X_train = dataframes['X_train'][lasso_features]
    X_test = dataframes['X_test'][lasso_features]
    return X_train, X_test, lasso_features

def get_lasso_scores(dataframes, features, thresholds):    
    lasso_fitting, importance = get_lasso_fitting(dataframes)
    scores = init_scores()
    for threshold in thresholds:
        X_train, X_test, lasso_features = get_lasso_features(dataframes, features, lasso_fitting, importance, threshold)
        for name, classifier, _, _ in CLASSIFIERS:
            print(threshold, name)
            parameters=FITTING_PARAMETERS[name]
            model = GridSearchCV(classifier, parameters, cv=2).fit(X_train, dataframes['Y_class_train'])
            estimator = model.best_estimator_
            y_pred = estimator.predict(X_test)
            scores[name] = append_scores(scores[name], dataframes['Y_class_test'], y_pred, estimator, lasso_features)
    return scores
