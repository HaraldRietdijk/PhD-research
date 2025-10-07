import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from steps.step_generic_code.general_variables.general_variables_all_shap import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS

def init_scores():
    scores = {}
    for name, _, _ , _ in CLASSIFIERS:
        scores[name]={'accuracy' : [], 'f1-score' : [], 'precision' : [], 'recall' : [], 'features' : [], 'coefficients' : []}
    return scores

def append_scores(scores, report, estimator, features):
    scores['accuracy'].append(report['accuracy'])
    scores['f1-score'].append(report['weighted avg']['f1-score'])
    scores['precision'].append(report['weighted avg']['precision'])
    scores['recall'].append(report['weighted avg']['recall'])
    scores['features'].append(list(features))
    if hasattr(estimator, 'coef_'):
        coefficients = list(estimator.coef_[0])
    else:
        coefficients = list(estimator.feature_importances_)
    scores['coefficients'].append(coefficients)
    return scores

def get_scores_for_method(method, dataframes, features):
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
            report = classification_report(dataframes['Y_class_test'], y_pred,output_dict=True)
            scores[name] = append_scores(scores[name], report, estimator, method_features)
    return scores

def get_filter_methods_scores(dataframes, features):
    # do anova and chi2 selecting all features, and then select based on coef.
    filter_method_scores = {}
    print('Getting scores for anova.')
    filter_method_scores['anova'] = get_scores_for_method(f_classif, dataframes, features)
    print('Getting scores for chi2.')
    filter_method_scores['chi2'] = get_scores_for_method(chi2, dataframes, features)
    return filter_method_scores
