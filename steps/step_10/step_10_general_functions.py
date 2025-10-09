from database.models.feature_selection_data import SELECTION_METHOD, METHOD_RESULTS, METHOD_RESULTS_FEATURES
from steps.step_10.step_10_plot import plot_average_per_method, plot_results_per_method

def store_model_result(app, method_id, run_id, model, idx, score_per_model, thresholds=None):
    nr_features = len(score_per_model['features'][idx])
    if thresholds:
        threshold = thresholds[idx]
    else:
        threshold = 0
    method_results = METHOD_RESULTS(method_id = method_id,
                                    run_id = run_id, 
                                    model = model, 
                                    nr_features = nr_features,
                                    threshold = threshold,
                                    accuracy = score_per_model['accuracy'][idx],
                                    f1_score = score_per_model['f1-score'][idx],
                                    precision = score_per_model['precision'][idx],
                                    recall = score_per_model['recall'][idx]
                                    )
    app.session.add(method_results)
    app.session.commit()
    return method_results.id

def store_features_for_result(app, features, coefficients, method_results_id):
    for idx, feature in enumerate(features):
        features = METHOD_RESULTS_FEATURES(result_id = method_results_id,
                                            feature = feature,
                                            coefficient = coefficients[idx]
                                            )
        app.session.add(features)
    app.session.commit()

def get_method_id(app, method):
    selection_method = app.session.query(SELECTION_METHOD).filter(SELECTION_METHOD.name==method).first()
    if not selection_method:
        selection_method = SELECTION_METHOD(name=method, type='new type')
        app.session.add(selection_method)
        app.session.commit()
    return selection_method.id

def save_method_results(app, scores, run_id, thresholds = None):
    for method, scores_per_method in scores.items():
        method_id = get_method_id(app, method)
        for model, score_per_model in scores_per_method.items():
            for idx in range(len(score_per_model['accuracy'])):
                method_results_id = store_model_result(app, method_id, run_id, model, 
                                                       idx, score_per_model, thresholds)
                store_features_for_result(app, score_per_model['features'][idx], 
                                          score_per_model['coefficients'][idx] , method_results_id)

def plot_and_save_results(app, folder, lasso_scores, run_id, thresholds=None):
    plot_results_per_method(folder, lasso_scores, run_id, thresholds=thresholds)
    save_method_results(app, lasso_scores, run_id, thresholds=thresholds)
    plot_average_per_method(app, folder, run_id, thresholds=thresholds)
