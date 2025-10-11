from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from database.models.feature_selection_data import SELECTION_METHOD, METHOD_RESULTS, METHOD_RESULTS_FEATURES
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
