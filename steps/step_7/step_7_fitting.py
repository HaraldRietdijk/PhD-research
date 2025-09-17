import time

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from steps.step_generic_code.general_variables.general_variables_dataset_clustering import CLASSIFIERS, FITTING_PARAMETERS
from steps.step_7.step_7_input_preparation import get_full_sets
from steps.step_7.step_7_storing_results import save_model_and_pickle,store_metrics

def get_scores(x, y, model):
    y_pred = model.predict(x)
    scores = {}
    df_confusion_matrix=pd.DataFrame(confusion_matrix(y, y_pred))
    scores['TN'] = int(df_confusion_matrix.iloc[0,0])
    scores['FP'] = int(df_confusion_matrix.iloc[0,1])
    scores['FN'] = int(df_confusion_matrix.iloc[1,0])
    scores['TP'] = int(df_confusion_matrix.iloc[1,1])
    scores['F1'] = float(f1_score(y, y_pred, average='micro'))
    scores['ACC'] = float(accuracy_score(y, y_pred))
    return scores

def fit_model (name, use_algorithm, X_train, Y_train):
    start = time.time()
    if name in ['RF','DT']:
        fitted_model = GridSearchCV(use_algorithm, FITTING_PARAMETERS[name],error_score=0.0, cv=5).fit(X_train, Y_train)
    else:
        fitted_model = GridSearchCV(use_algorithm, FITTING_PARAMETERS[name], cv=5).fit(X_train, Y_train)
    end = time.time()
    return fitted_model, (end-start)

def fit_clusters(cluster_sets):
    fitting_times_all_clusters = {}
    for cluster_code, sets in cluster_sets.items():
        fitting_times_all_models = {}
        for name, classifier, normalized in CLASSIFIERS: 
            print(cluster_code, name)
            if normalized:
                x_train_set = sets['X_train_s']
            else:
                x_train_set = sets['X_train']
            fitted_model, fitting_time = fit_model(name, classifier, x_train_set, sets['Y_train'])
            fitting_times_all_models[name] = (fitted_model, fitting_time)
        fitting_times_all_clusters[cluster_code] = fitting_times_all_models
    return fitting_times_all_clusters
    
def make_prediction(app, folder, cluster_dfs, fitting_times_all_clusters, row, run_id):
    classifier_normalized = {name : normalized for (name, _, normalized) in CLASSIFIERS}
    for cluster_name, cluster in cluster_dfs.items():
        for classifier_name, fitting_tuple in fitting_times_all_clusters[cluster_name].items():
            cluster_id = 'fg_' + str(row['id']) + 'cg_' + cluster_name
            model_id = save_model_and_pickle(app, folder, cluster_id, classifier_name, fitting_tuple)
            variable_threshold = not(row['dataset'] == 'vfc')
            X, X_scaled, Y = get_full_sets(cluster, variable_threshold)
            if classifier_normalized[classifier_name]:
                y_pred = fitting_tuple[0].predict(X_scaled)
            else:
                y_pred = fitting_tuple[0].predict(X)
            store_metrics(app, row['id'], cluster_name, model_id, Y, y_pred, run_id)             

