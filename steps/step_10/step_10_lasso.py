import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import pandas as pd

from steps.step_10.step_10_general_functions import append_scores_for_features, init_scores, save_feature_ranking, save_method_results
from steps.step_generic_code.general_functions import complete_run, get_run_id
from steps.step_generic_code.general_variables.general_variables_all_shap import CLASSIFIERS, LASSO_THRESHOLDS

def get_lasso_fitting(dataframes):
    params = {"alpha":[0.0001]}
    lasso_fitting = GridSearchCV(Lasso(), param_grid=params, cv=5).fit(dataframes['X'],dataframes['Y_class']).best_estimator_
    importance = pd.DataFrame([abs(coefficient) for coefficient in lasso_fitting.coef_] ,
                          columns=['coefficient'])
    importance['normalized'] = importance/importance.max()
    importance['features'] = pd.DataFrame(feature for feature in lasso_fitting.feature_names_in_)
    return lasso_fitting, importance

def get_coefficient_threshold(importance, threshold):
    valid_coefficients = importance.loc[importance['normalized']>=threshold]
    return valid_coefficients['coefficient'].min()

def get_lasso_features(app, dataframes, features, thresholds, source, steps):
    lasso_fitting, importance = get_lasso_fitting(dataframes)
    save_feature_ranking(app, 'LASSO', features, importance, source)
    lasso_features = []
    if steps:
        for aantal in range(1,len(features)):
            select_from_model = SelectFromModel(lasso_fitting, threshold=-np.inf, max_features=aantal)
            lasso_features.append(select_from_model.fit(dataframes['X'],dataframes['Y_class']).get_feature_names_out(features))
    else:
        for threshold in thresholds:
            coefficient_threshold = get_coefficient_threshold(importance, threshold)
            select_from_model = SelectFromModel(lasso_fitting, threshold=coefficient_threshold)
            lasso_features.append(select_from_model.fit(dataframes['X'],dataframes['Y_class']).get_feature_names_out(features))
    return lasso_features

def get_lasso_scores(lasso_features_selection, dataframes, thresholds):    
    scores = init_scores()
    for idx, lasso_features in enumerate(lasso_features_selection):
        for name, classifier, _, _ in CLASSIFIERS:
            print(thresholds[idx], name)
            scores[name] = append_scores_for_features(scores, name, classifier, dataframes, lasso_features)
    return scores

def get_lasso_scores_steps(lasso_features_selection, dataframes, thresholds):    
    scores = init_scores()
    for lasso_features in lasso_features_selection:
        for name, classifier, _, _ in CLASSIFIERS:
            print(len(lasso_features),name)
            scores[name] = append_scores_for_features(scores, name, classifier, dataframes, lasso_features)
    return scores

def do_lasso(app, dataframes, features, source='NS', steps=False):
    thresholds = LASSO_THRESHOLDS[source]
    lasso_features = get_lasso_features(app, dataframes, features, thresholds, source, steps)
    for i in range(30):
        if steps:
            text = 'Starting lasso run with steps: '
            method = 'LASSO_step' 
        else: 
            text = 'Starting lasso run: '
            method = 'LASSO'
        print(text,str(i+1))
        run_id = get_run_id(app,"Feature Selection LASSO", 'test', 10, source)
        lasso_scores = {}
        if steps:
            lasso_scores[method] = get_lasso_scores_steps(lasso_features, dataframes, thresholds)
            save_method_results(app, lasso_scores, run_id, 'embedded')
        else:
            lasso_scores[method] = get_lasso_scores(lasso_features, dataframes, thresholds)
            save_method_results(app, lasso_scores, run_id, 'embedded', thresholds=thresholds)
        complete_run(app, run_id)
