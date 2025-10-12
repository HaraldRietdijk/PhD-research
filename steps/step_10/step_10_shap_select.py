from sklearn.model_selection import GridSearchCV
from shap_select import shap_select

from steps.step_10.step_10_general_functions import append_scores, init_scores, save_method_results
from steps.step_generic_code.general_functions import complete_run, get_run_id
from steps.step_generic_code.general_variables.general_variables_all_shap import CLASSIFIERS, FITTING_PARAMETERS, SHAP_THRESHOLDS, SEEDS

def get_shap_significance(estimator, dataframes, features, threshold):
    return shap_select(
        tree_model=estimator,
        validation_df=dataframes['X'],
        target=dataframes['Y_class'],
        feature_names=features,
        task="binary",
        threshold=threshold,
        return_extended_data=False
    )

def get_shap_features(shap_significance, threshold):
    selected = shap_significance.loc[shap_significance["stat.significance"] < threshold, "feature name"].tolist()
    if len(selected) == 0:
        top_k = 1
        cand = shap_significance.sort_values("t-value", ascending=False)
        selected = cand["feature name"].head(top_k).tolist()
    return selected

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

def get_shap_scores(dataframes, shap_features, thresholds):
    scores = init_scores()
    for idx, threshold in enumerate(thresholds):
        for name, classifier, _, _ in CLASSIFIERS:
            print(threshold, name)
            shap_features = shap_features[name][idx]
            X_train = dataframes['X_train'][shap_features]
            X_test = dataframes['X_test'][shap_features]
            estimator = get_fitted_model(name, classifier, X_train, dataframes['Y_class_train'])
            y_pred = estimator.predict(X_test)
            scores[name] = append_scores(scores[name], dataframes['Y_class_test'], y_pred, estimator, shap_features)
    return scores

def get_base_features(app, dataframes, features, thresholds):
    # Get the feature ranking based on the base model 
    # and store the score for the base model using all 44 features
    # Base model used to be able to compare the results of the shap-fitted models to the model with all features
    base_features = {}
    base_scores = init_scores()
    run_id = get_run_id(app,"Feature Selection shap base", 'test', 10, 'NS')
    for name, classifier, _, _ in CLASSIFIERS:
        base_model = get_fitted_model(name, classifier, dataframes['X_train'], dataframes['Y_class_train'])
        y_pred = base_model.predict(dataframes['X_test'])
        base_scores[name] = append_scores(base_scores[name], dataframes['Y_class_test'], y_pred, base_model, features)
        base_features[name] = []
        shap_significance = get_shap_significance(base_model, dataframes, features, threshold=0.05)
        for threshold in thresholds:
            print('base fitting ', name,' : ', threshold)
            shap_features = get_shap_features(shap_significance, threshold)
            base_features[name].append(shap_features)
    shap_base_score = {'shapbase' : base_scores}
    save_method_results(app, shap_base_score, run_id, 'base', thresholds=thresholds)
    complete_run(app, run_id)
    return base_features

def do_shap_select(app, dataframes, features):
    shap_scores = {}
    thresholds = SHAP_THRESHOLDS
    shap_features = get_base_features(app, dataframes, features, thresholds,)
    for i in range(30):
        print('Starting shap-select run: ',str(i+1))
        run_id = get_run_id(app,"Feature Selection shap", 'test', 10, 'NS')
        shap_scores['shap'] = get_shap_scores(dataframes, shap_features, thresholds)
        save_method_results(app, shap_scores, run_id, 'embedded', thresholds=thresholds)
        complete_run(app, run_id)