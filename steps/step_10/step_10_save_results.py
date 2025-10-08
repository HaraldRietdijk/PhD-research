from database.models.feature_selection_data import SELECTION_METHOD, METHOD_RESULTS, METHOD_RESULTS_FEATURES

def store_model_result(app, method_id, run_id, accuracy, model, idx, scores_per_model):
    method_results = METHOD_RESULTS(method_id = method_id,
                                run_id = run_id, 
                                model = model, 
                                nr_features = idx+1,
                                accuracy = accuracy,
                                f1_score = scores_per_model['f1-score'][idx],
                                precision = scores_per_model['precision'][idx],
                                recall = scores_per_model['recall'][idx]
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

def save_method_results(app, scores, run_id):
    for method, scores_per_method in scores.items():
        method_id = get_method_id(app, method)
        for model, score_per_model in scores_per_method.items():
            for idx, accuracy in enumerate(score_per_model['accuracy']):
                method_results_id = store_model_result(app, method_id, run_id, accuracy, model, idx, score_per_model)
                store_features_for_result(app, score_per_model['features'][idx], score_per_model['coefficients'][idx] , method_results_id)
