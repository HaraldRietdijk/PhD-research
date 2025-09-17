import logging
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score

from database.models.patient_data import RFE_RESULTS, RFE_RESULTS_FEATURES
from steps.step_generic_code.general_variables.general_variables_all_rtw_1 import CLASSIFIERS
from steps.step_generic_code.general_variables.general_variables_all_rtw_1 import CLASSIFIERS_FOR_TREE_OPT
from steps.step_8.step_8_generic_RFE import do_RFE, store_RFE_results, store_features_ranking,get_reduced_frames,\
                                            get_Y_frames, store_score_result, get_dropped_feature,\
                                            get_sorted_frames, get_results_list

def get_classifier_scores(Y, pred, probability, Y_proba):
    scores = {}
    scores['f1_score'] = f1_score(Y, pred, average='macro')
    scores['accuracy'] = accuracy_score(Y, pred)
    scores['roc_auc'] = 0
    if probability:
        if len(Y_proba[0])==2:
            Y_proba = Y_proba[:,1]        
        scores['roc_auc'] = roc_auc_score(Y, Y_proba, average='macro', multi_class='ovo')
    scores['precision'] = precision_score(Y, pred, average='macro')
    scores['recall'] = recall_score(Y, pred, average='macro')
    return scores

def strore_score_for_set(app, Y_frames, rfe_result_id, test_or_train, probability):
    Y_set = 'Y_' + test_or_train
    pred = 'pred_' + test_or_train
    proba = 'proba_' + test_or_train
    scores = get_classifier_scores(Y_frames[Y_set], Y_frames[pred], probability, Y_frames[proba])
    store_score_result(app, 'f1_score', scores['f1_score'], rfe_result_id, test_or_train)
    store_score_result(app, 'accuracy', scores['accuracy'], rfe_result_id, test_or_train)
    store_score_result(app, 'precision', scores['precision'], rfe_result_id, test_or_train)
    store_score_result(app, 'recall', scores['recall'], rfe_result_id, test_or_train)
    if probability:
        store_score_result(app, 'area_under_curve', scores['roc_auc'], rfe_result_id, test_or_train)

def check_probability(name):
    probability = True
    if name in ['PAC','PER','RIC','SGD','SGDOC','NCC','PAC1','PER1','RIC1','SGD1','SGDOC1','NCC1','LSVC','LSVC1']:
        probability = False
    return probability

def store_score_classifier(app, fit, features, dataframes, rfe_result_id, name):
    probability = check_probability(name)
    Y_frames = get_Y_frames(fit, features, dataframes['X'], dataframes['Y_class'], probability)
    strore_score_for_set(app, Y_frames, rfe_result_id, 'test', probability)
    strore_score_for_set(app, Y_frames, rfe_result_id, 'train', probability)

def get_score(fit, features, dataframes, name):
    probability = check_probability(name)
    Y_frames = get_Y_frames(fit, features, dataframes['X'], dataframes['Y_class'], probability)
    return get_classifier_scores(Y_frames['Y_test'], Y_frames['pred_test'], probability, Y_frames['proba_test'])

def do_RFE_for_nr_of_features(app, reduced_dataframes, name, nr_of_classes, nr_of_features, model, usefitted, last, is_max):
    features = reduced_dataframes['X'].columns.tolist()
    fit = do_RFE(reduced_dataframes['X_train'], reduced_dataframes['Y_class_train'], model, usefitted, nr_of_features)
    rfe_results_id = store_RFE_results(app, fit, name, 1, nr_of_classes)
    store_features_ranking(app, fit, features, rfe_results_id, last=last, is_max=is_max)
    store_score_classifier(app, fit, features, reduced_dataframes, rfe_results_id, name)
    return fit, features

def RFE_classifiers(app, dataframes, nr_of_classes, fitted_models):
    for name, _, usefitted, is_max in CLASSIFIERS:
        model = fitted_models[name]
        print('Step 8: RFE for estimator: ' + name)
        logging.info(name)
        reduced_dataframes = dataframes
        for nr_of_features in range(40, 3, -2 ):
            fit, features = do_RFE_for_nr_of_features(app, reduced_dataframes, name, nr_of_classes, nr_of_features, model, usefitted, 
                                                           last=(nr_of_features==4), is_max=(nr_of_features==is_max))
            reduced_dataframes = get_reduced_frames(fit, features, reduced_dataframes)

def concat_feature_ranking_line(feature_ranking, last_feature, last_scores, rank, adjusted_rank, last_importance):
    return pd.concat([feature_ranking,pd.DataFrame({'feature':[last_feature], 'scores':[last_scores], 'ranking':[rank], 
                                                    'adjusted_rank': [adjusted_rank], 'importance': [last_importance]})])

def add_ranking_line(feature_ranking, scores, last_scores, nr_of_features, last_feature, last_importance):
    rank = nr_of_features + 1
    adjusted_rank = 2    
    scoring = last_scores['accuracy'] - scores['accuracy']
    if scoring<0:
        adjusted_rank += (rank/40)
    elif scoring>0:
        adjusted_rank -= ((50-rank)/40)
    return concat_feature_ranking_line(feature_ranking, last_feature, last_scores, rank, adjusted_rank, last_importance)

def do_optimization(model, dataframes, feature_ranking, probability):
    sorted_features = feature_ranking['feature'].to_list()
    reduced_dataframes = get_sorted_frames(sorted_features, dataframes)
    new_feature_ranking = pd.DataFrame(columns=['feature', 'scores', 'ranking', 'adjusted_rank', 'importance'])
    features = sorted_features
    max_accuracy = 0
    for nr_of_features in range(44, 0, -1):
        fit = model.fit(reduced_dataframes['X_train'], reduced_dataframes['Y_class_train'])
        pred = fit.predict(reduced_dataframes['X_test'])
        if probability:
            Y_proba = fit.predict_proba(reduced_dataframes['X_test'])
        else:
            Y_proba = None
        scores = get_classifier_scores(reduced_dataframes['Y_class_test'], pred, probability, Y_proba)
        max_accuracy = max(max_accuracy, scores['accuracy'])
        if (nr_of_features<44):
            new_feature_ranking = add_ranking_line(new_feature_ranking, scores, last_scores, nr_of_features, last_feature, last_importance)
        last_feature = features[-1]
        importance_list, _ = get_results_list(fit, use_estimator=False)
        last_importance = importance_list[-1]
        features = sorted_features[0:nr_of_features-1]
        reduced_dataframes = get_sorted_frames(features, dataframes)
        last_scores = scores
    new_feature_ranking = concat_feature_ranking_line(new_feature_ranking, last_feature, last_scores, 1, 1, last_importance)
    return new_feature_ranking.sort_values(by=['adjusted_rank'], ascending=[True]), max_accuracy

def get_RFE_ranking(app, dataframes, model, name):
        reduced_dataframes = dataframes
        feature_ranking = pd.DataFrame(columns=['feature', 'scores', 'ranking','adjusted_rank', 'importance'])
        for nr_of_features in range(44, 0, -1 ):
            fit, features = do_RFE_for_nr_of_features(app, reduced_dataframes, name, 2, nr_of_features, model, True, 
                                                           last=(nr_of_features==1), is_max=(nr_of_features==100))
            reduced_dataframes = get_reduced_frames(fit, features, reduced_dataframes)
            last_feature, feature_scores = get_dropped_feature(fit, features, last=(nr_of_features==1))
            scores = get_score(fit, features, dataframes, name)
            if (nr_of_features<44):
                last_importance = last_feature_scores[last_feature[0][1]]
                feature_ranking = add_ranking_line(feature_ranking, scores, last_scores, nr_of_features, last_feature[0][1], last_importance)
            last_feature_scores = feature_scores    
            last_scores = scores
        last_importance = feature_scores[last_feature[1][1]]
        feature_ranking = concat_feature_ranking_line(feature_ranking, last_feature[1][1], last_scores, 1, 1, last_importance)
        return feature_ranking.sort_values(by=['adjusted_rank'], ascending=[True])

def store_final_ranking(app, feature_ranking, name, probability):
    def store_ranking(row):
        rfe_result = RFE_RESULTS(estimator = name, nr_classes=2, nr_features=44, nr_selected=row['ranking'], nr_runs=1, nr_splits=1)
        app.session.add(rfe_result)
        app.session.commit()
        store_score_result(app, 'accuracy', row['scores']['accuracy'], rfe_result.id, 'optim')
        store_score_result(app, 'f1_score', row['scores']['f1_score'], rfe_result.id, 'optim')
        if probability:
            store_score_result(app, 'area_under_curve', row['scores']['roc_auc'], rfe_result.id, 'optim')
        store_score_result(app, 'precision', row['scores']['precision'], rfe_result.id, 'optim')
        store_score_result(app, 'recall', row['scores']['recall'], rfe_result.id, 'optim')
        rfe_result_feature = RFE_RESULTS_FEATURES(rfe_results_id=rfe_result.id, feature=row['feature'], ranking=row['ranking'], importance=row['importance'])
        app.session.add(rfe_result_feature)
        app.session.commit()

    feature_ranking.apply(store_ranking, axis=1)

def classifiers_optimization(app, name, dataframes, model, probability):
    logging.info(name)
    feature_ranking = get_RFE_ranking(app, dataframes, model, name)
    max_accuracy = -1
    new_accuracy = 0
    optimal_count = 0
    while ((max_accuracy <= new_accuracy) or (optimal_count<10)):
        if max_accuracy < new_accuracy:
            max_accuracy = new_accuracy
            new_optimal = True
        new_feature_ranking, new_accuracy = do_optimization(model, dataframes, feature_ranking, probability)
        print('this itteration:', max_accuracy, new_accuracy, optimal_count)
        if max_accuracy <= new_accuracy:
            optimal_feature_ranking = new_feature_ranking
            if optimal_count>5:
                optimal_count = 4
        else:
            if not new_optimal:
                optimal_count += 1
            else:
                new_optimal = False
                optimal_count = 0
        feature_ranking = new_feature_ranking
    store_final_ranking(app, optimal_feature_ranking, name, probability)

def RFE_classifiers_optimize(app, dataframes):
    for name, model in CLASSIFIERS_FOR_TREE_OPT:
        print('Step 8: RFE optimization for estimator: ' + name)
        probability = check_probability(name)
        classifiers_optimization(app, name, dataframes, model, probability)

def RFE_classifiers_optimize_top(app, dataframes, fitted_models):
    for name, _, _, _ in CLASSIFIERS:
        model = fitted_models[name]
        print('Step 8: RFE optimization for best estimator: ' + name)
        probability = check_probability(name)
        classifiers_optimization(app, name, dataframes, model, probability)
