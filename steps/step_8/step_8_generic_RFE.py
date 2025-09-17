import pandas as pd

from sqlalchemy import and_

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from database.models.patient_data import RFE_RESULTS, RFE_RESULTS_FEATURES, RFE_SCORING, RFE_SCORING_TYPES

def store_RFE_results(app, fit, name, repeat, nr_of_classes, nr_splits=1, split_group=None):
    nr_features = len(fit.ranking_)
    nr_selected= fit.n_features_
    rfe_result = RFE_RESULTS(estimator = name, nr_classes=nr_of_classes, 
                             nr_features=nr_features, nr_selected=nr_selected, nr_runs=repeat,
                             nr_splits=nr_splits, split_group=split_group)
    app.session.add(rfe_result)
    app.session.commit()
    return rfe_result.id

def get_results_list(fit, use_estimator = True):
    if use_estimator:
        estimator = fit.estimator_
    else:
        estimator = fit
    if hasattr(estimator, 'coef_'):
        coefficients = True
        results_list = estimator.coef_.tolist()
        if isinstance(results_list[0], list):
            temp_list = [sum(i)/len(results_list) for i in zip(*results_list)]
            results_list = temp_list
    else:
        coefficients = False
        results_list = estimator.feature_importances_.tolist()
    return results_list, coefficients

def check_order_dropped_features(dropped_features):
    feature1 = dropped_features[0]
    feature2 = dropped_features[1]
    new_dropped_features=[]
    if (feature1[0]<feature2[0]):
        new_dropped_features=[feature2, feature1]
    else:
        new_dropped_features = dropped_features
    return new_dropped_features

def get_dropped_feature(fit, features, last):
    dropped_features =[]
    feature_scores ={}
    results_list, coefficients = get_results_list(fit)
    # results_list = fit.estimator_.feature_importances_.tolist()
    index = 0
    for rank, feature in zip(fit.ranking_, features):
        if rank>1 or last:
            rank += fit.n_features_ - 1
            dropped_features.append((rank, feature))
            if rank==1:
                feature_scores[feature] = results_list[index]
        else:
            feature_scores[feature] = results_list[index]
            index += 1
    if len(dropped_features)==2:
        dropped_features = check_order_dropped_features(dropped_features)
    return dropped_features, feature_scores

def store_features_ranking(app, fit, features, rfe_result_id, last, is_max):
    results_list, coefficients = get_results_list(fit)
    index = 0
    for rank, feature in zip(fit.ranking_, features):
        if rank>1:
            rank += fit.n_features_ - 1
            rfe_result = RFE_RESULTS_FEATURES(rfe_results_id=rfe_result_id, feature=feature, ranking=rank)
            app.session.add(rfe_result)
        else:
        # elif last or is_max:
            if coefficients:
                coefficient = results_list[index]
                importance = None
            else:
                coefficient = None
                importance = results_list[index]
            index += 1
            ranking = 99
            if fit.n_features_ == 1:
                ranking = 1
            rfe_result = RFE_RESULTS_FEATURES(rfe_results_id=rfe_result_id, feature=feature, ranking=ranking, coefficient=coefficient, importance=importance)
            app.session.add(rfe_result)
    app.session.commit()

def store_score_result(app, name, value, rfe_result_id, test_or_train):
    scoring_type = app.session.query(RFE_SCORING_TYPES).filter(RFE_SCORING_TYPES.name==name).first()
    if not scoring_type:
        scoring_type = RFE_SCORING_TYPES(name=name)
        app.session.add(scoring_type)
        app.session.commit()
    rfe_scoring = RFE_SCORING(rfe_results_id = rfe_result_id, test_or_train_data = test_or_train, 
                              scoring_type_id = scoring_type.id, value = value)
    app.session.add(rfe_scoring)
    app.session.commit()

def get_reduced_X(fit, features, X):
    X_sel = pd.DataFrame()
    for rank, feature in zip(fit.ranking_, features):
        if rank==1:
            X_sel[feature]=X[feature]
    return X_sel

def get_sorted_X(features, X):
    X_sel = pd.DataFrame()
    for feature in features:
        X_sel[feature]=X[feature]
    return X_sel

def get_Y_frames(fit, features, X, Y, probability):
    Y_frames = {}
    X_train, X_test, Y_frames['Y_train'], Y_frames['Y_test'] = train_test_split(X,Y, test_size=0.3, random_state=42)
    X_train = get_reduced_X(fit, features, X_train)
    X_test = get_reduced_X(fit, features, X_test)
    Y_frames['pred_test'] = fit.estimator_.predict(X_test)
    Y_frames['pred_train'] = fit.estimator_.predict(X_train)
    Y_frames['proba_test'] = None
    Y_frames['proba_train'] = None
    if probability:
        Y_frames['proba_test'] = fit.estimator_.predict_proba(X_test)
        Y_frames['proba_train'] = fit.estimator_.predict_proba(X_train)
    return Y_frames

def get_sorted_frames(features, dataframes):
    sorted_frames = {}
    sorted_frames['X'] = get_sorted_X(features, dataframes['X'])
    sorted_frames['X_train'] = get_sorted_X(features, dataframes['X_train'])
    sorted_frames['X_test'] = get_sorted_X(features, dataframes['X_test'])
    sorted_frames['Y'] = dataframes['Y']
    sorted_frames['Y_test'] = dataframes['Y_test']
    sorted_frames['Y_train'] = dataframes['Y_train']
    sorted_frames['Y_class'] = dataframes['Y_class']
    sorted_frames['Y_class_test'] = dataframes['Y_class_test']
    sorted_frames['Y_class_train'] = dataframes['Y_class_train']
    return sorted_frames

def get_reduced_frames(fit,features, dataframes):
    reduced_frames = {}
    reduced_frames['X'] = get_reduced_X(fit, features, dataframes['X'])
    reduced_frames['X_train'] = get_reduced_X(fit, features, dataframes['X_train'])
    reduced_frames['X_test'] = get_reduced_X(fit, features, dataframes['X_test'])
    reduced_frames['Y'] = dataframes['Y']
    reduced_frames['Y_test'] = dataframes['Y_test']
    reduced_frames['Y_train'] = dataframes['Y_train']
    reduced_frames['Y_class'] = dataframes['Y_class']
    reduced_frames['Y_class_test'] = dataframes['Y_class_test']
    reduced_frames['Y_class_train'] = dataframes['Y_class_train']
    return reduced_frames

def do_RFE(X, Y, model, usefitted, nr_of_features):
    if usefitted:
        model = model.fit(X,Y)
    rfe = RFE(estimator=model, n_features_to_select=nr_of_features)
    fit = rfe.fit(X, Y)
    return fit