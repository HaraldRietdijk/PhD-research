import logging
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from database.models.feature_selection_data import SELECTION_METHOD, METHOD_RESULTS, METHOD_RESULTS_FEATURES
from steps.step_generic_code.general_variables.general_variables_all_shap import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS
      

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

def fit_models(X_train, Y_train):
    print("Step 10 CS: Fitting models")
    fitted_models = {}
    for name, classifier, _, _ in CLASSIFIERS:
        print("Model: ", name)
        parameters=FITTING_PARAMETERS[name]
        log_text="Model {m}".format(m=name)
        logging.info(log_text)
        if name in ['OCS','SGDOC']:
            fitted_models[name] = GridSearchCV(classifier, parameters, scoring='accuracy', cv=2).fit(X_train, Y_train).best_estimator_
        else:
            newDF = X_train.copy()
            newDF.replace(np.nan, 'error', inplace =True)
            newDF.to_csv('newDF_train.csv', sep=',')
            fitted_models[name] = GridSearchCV(classifier, parameters, cv=2).fit(X_train, Y_train).best_estimator_
    return fitted_models

def get_scores_for_prediction(fitted_models, X_set, Y_set):
    print("Step 10 CS: geting scores")
    models_scoring = {}
    for name, fitted_model in fitted_models.items():
        model_score = {}
        Y_pred = fitted_model.predict(X_set)
        model_score['accuracy_score'] = accuracy_score(Y_set,Y_pred)
        model_score['f1_score_micro'] = f1_score(Y_set,Y_pred,average='micro')
        model_score['f1_score_macro'] = f1_score(Y_set,Y_pred,average='macro')
        if name not in ['PAC','PER','RIC','SGD','PAC1','PER1','RIC1','SGD1','SGDOC','NCC','LSVC','LSVC1']:
            Y_pred_prob = fitted_model.predict_proba(X_set)
            if len(Y_pred_prob[0])==2:
                Y_pred_prob = Y_pred_prob[:,1]
            model_score['roc_auc_score_micro'] = roc_auc_score(Y_set,Y_pred_prob,average='micro',multi_class='ovr')
            model_score['roc_auc_score_macro'] = roc_auc_score(Y_set,Y_pred_prob,average='macro',multi_class='ovo')
        models_scoring[name] = model_score
    return models_scoring

def classifier_selection(app, folder, dataframes, nr_of_classes):
    fitted_models = fit_models(dataframes['X_train'], dataframes['Y_class_train'])
    models_scoring = {}
    models_scoring['Test'] = get_scores_for_prediction(fitted_models, dataframes['X_test'], dataframes['Y_class_test'])
    models_scoring['Train'] = get_scores_for_prediction(fitted_models, dataframes['X_train'], dataframes['Y_class_train'])
    # save_pickle_and_metrics(app, folder, fitted_models, models_scoring, nr_of_classes, ALGORITHM_PARAMETERS)
    return fitted_models
