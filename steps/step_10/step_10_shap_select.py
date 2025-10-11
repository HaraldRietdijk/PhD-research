import random
import numpy as np

from sklearn.model_selection import GridSearchCV
from shap_select import shap_select

from steps.step_10.step_10_general_functions import append_scores, init_scores, save_method_results
from steps.step_generic_code.general_functions import complete_run, get_run_id
from steps.step_generic_code.general_variables.general_variables_all_shap import CLASSIFIERS, FITTING_PARAMETERS, SHAP_THRESHOLDS, SEEDS

def get_shap_features(estimator, dataframes, features, threshold):
    results = shap_select(
        tree_model=estimator,
        validation_df=dataframes['X'],
        target=dataframes['Y_class'],
        feature_names=features,
        task="binary",
        threshold=threshold,
        return_extended_data=True
    )
    significance_df, shap_values_df = results
    selected = significance_df.loc[significance_df["selected"] == 1, "feature name"].tolist()
    if len(selected) == 0:
        top_k = 2
        # cand = significance_df[significance_df["t-value"] > 0].sort_values("t-value", ascending=False)
        cand = significance_df.sort_values("t-value", ascending=False)
        selected = cand["feature name"].head(top_k).tolist()
    X_train = dataframes['X_train'][selected]
    X_test = dataframes['X_test'][selected]
    return X_train, X_test, selected

def get_fitted_model(name, classifier, X, Y):
    parameters = FITTING_PARAMETERS[name]
    fitted = False
    while (not fitted):
        try:
            model = GridSearchCV(classifier, parameters, cv=2).fit(X, Y)
            fitted = True
        except Exception as e:
            print(e)
            nr_used_features = len(X.columns.tolist())
            print('Failed to fit ', name, 'on ', nr_used_features,' features, let us try again')
    return model.best_estimator_

def get_shap_scores(dataframes, features, thresholds):
    scores = init_scores()
    for threshold in thresholds:
        for name, classifier, _, _ in CLASSIFIERS:
            print(threshold, name)
            estimator = get_fitted_model(name, classifier, dataframes['X'], dataframes['Y_class'])
            X_train, X_test, shap_features = get_shap_features(estimator, dataframes, features, threshold)
            estimator = get_fitted_model(name, classifier, X_train, dataframes['Y_class_train'])
            y_pred = estimator.predict(X_test)
            scores[name] = append_scores(scores[name], dataframes['Y_class_test'], y_pred, estimator, shap_features)
    return scores

def do_shap_select(app, dataframes, features):
    shap_scores = {}
    for i in range(2):
        for seed in SEEDS:
            np.random.seed(seed)
            random.seed(seed)
            print('Starting shap-select run: ',str(i+1), ' Seed: ',str(seed))
            run_id = get_run_id(app,"Feature Selection shap", 'test', 10, 'NS')
            thresholds = SHAP_THRESHOLDS
            shap_scores['shapselect'] = get_shap_scores(dataframes, features, thresholds)
            save_method_results(app, shap_scores, run_id, thresholds=thresholds)
            complete_run(app, run_id)

