import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif  
from sklearn.model_selection import GridSearchCV
from steps.step_10.step_10_plot import plot_average_per_method, plot_results_per_method
from steps.step_10.step_10_save_results import save_method_results
from steps.step_generic_code.general_functions import complete_run, get_run_id
from steps.step_generic_code.general_variables.general_variables_all_shap import FITTING_PARAMETERS, CLASSIFIERS

def init_scores():
    scores = {}
    for name, _, _ , _ in CLASSIFIERS:
        scores[name]={'accuracy' : [], 'f1-score' : [], 'precision' : [], 'recall' : [], 'features' : [], 'coefficients' : []}
    return scores

def append_scores(scores, Y, Y_pred, estimator, features):
    scores['accuracy'].append(accuracy_score(Y, Y_pred))
    scores['f1-score'].append(f1_score(Y,Y_pred, average='macro'))
    scores['precision'].append(precision_score(Y, Y_pred, average='macro'))
    scores['recall'].append(recall_score(Y, Y_pred, average='macro'))
    scores['features'].append(list(features))
    if hasattr(estimator, 'coef_'):
        coefficients = list(estimator.coef_[0])
    else:
        coefficients = list(estimator.feature_importances_)
    scores['coefficients'].append(coefficients)
    return scores

def get_scores_for_filter_method(method, dataframes, features):
    scores = init_scores()
    for k in range(1,dataframes['X_train'].shape[1]):
        select_method = SelectKBest(method,k=k)
        method_features = select_method.fit(dataframes['X_train'],dataframes['Y_class_train']).get_feature_names_out(features)
        X_train = dataframes['X_train'][method_features]
        X_test = dataframes['X_test'][method_features]
        for name, classifier, _, _ in CLASSIFIERS:
            print(k,name)
            parameters=FITTING_PARAMETERS[name]
            model = GridSearchCV(classifier, parameters, cv=2).fit(X_train, dataframes['Y_class_train'])
            estimator = model.best_estimator_
            y_pred = estimator.predict(X_test)
            scores[name] = append_scores(scores[name], dataframes['Y_class_test'], y_pred, estimator, method_features)
    return scores

def get_filter_methods_scores(dataframes, features):
    # do anova and chi2 selecting all features, and then select based on coef.
    filter_method_scores = {}
    print('Getting scores for anova.')
    filter_method_scores['anova'] = get_scores_for_filter_method(f_classif, dataframes, features)
    print('Getting scores for chi2.')
    filter_method_scores['chi2'] = get_scores_for_filter_method(chi2, dataframes, features)
    return filter_method_scores

def do_filter_methods(app, dataframes, features, folder):
    # run_id=398 # use for reprinting specific run
    for i in range(5):
        print('Starting filter run: ',str(i+1))
        run_id = get_run_id(app,"Feature Selection filter", 'test', 10, 'NS')
        filter_methods_scores = get_filter_methods_scores(dataframes, features)
        plot_results_per_method(folder, filter_methods_scores, run_id)
        save_method_results(app, filter_methods_scores, run_id)
        plot_average_per_method(app, folder, run_id)
        complete_run(app, run_id)

