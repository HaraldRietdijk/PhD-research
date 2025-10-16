from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import pandas as pd

from steps.step_10.step_10_general_functions import append_scores_for_features, init_scores, save_method_results
from steps.step_generic_code.general_functions import complete_run, get_run_id
from steps.step_generic_code.general_variables.general_variables_all_shap import CLASSIFIERS, LASSO_THRESHOLDS

def get_lasso_fitting(dataframes):
    params = {"alpha":[0.0001]}
    lasso_fitting = GridSearchCV(Lasso(), param_grid=params, cv=5).fit(dataframes['X'],dataframes['Y_class']).best_estimator_
    importance = pd.DataFrame([abs(coefficient) for coefficient in lasso_fitting.coef_] ,
                          columns=['coefficient'])
    importance['normalized'] = importance/importance.max()
    return lasso_fitting, importance

def get_coefficient_threshold(importance, threshold):
    valid_coefficients = importance.loc[importance['normalized']>=threshold]
    return valid_coefficients['coefficient'].min()

def get_lasso_features(dataframes, features, thresholds):
    lasso_fitting, importance = get_lasso_fitting(dataframes)
    save_lasso = pd.DataFrame()
    save_lasso['features']=features
    save_lasso['score']=importance['normalized']
    save_lasso.to_csv('lasso_features.csv', index=False)#TO DO: should be put in database.
    lasso_features = []
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

def do_lasso(app, dataframes, features):
    thresholds = LASSO_THRESHOLDS
    lasso_features = get_lasso_features(dataframes, features, thresholds)
    print(lasso_features)
    for i in range(30):
        print('Starting lasso run: ',str(i+1))
        run_id = get_run_id(app,"Feature Selection LASSO", 'test', 10, 'NS')
        lasso_scores = {}
        lasso_scores['LASSO'] = get_lasso_scores(lasso_features, dataframes, thresholds)
        save_method_results(app, lasso_scores, run_id, 'embedded', thresholds=thresholds)
        complete_run(app, run_id)
        