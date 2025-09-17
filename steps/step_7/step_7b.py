import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run
from steps.step_7.step_7_input_preparation import get_feature_groups_for_prediction, get_models, get_dataframes_per_cluster,\
                                                  add_pickled_models, get_cluster_per_treatment, get_dataframe_per_treatment, get_treatments_per_cluster
from steps.step_7.step_7_storing_results import store_metrics_per_cluster

def get_predictions(app, folder, feature_groups, dataframes, run_id):
    for feature_group in feature_groups:
        print(feature_group)
        fitted_models = get_models(app, feature_group)
        fitted_models = add_pickled_models(folder, fitted_models)
        # treatment_cluster = get_cluster_per_treatment(app, feature_group)
        cluster_treatment = get_treatments_per_cluster(app, feature_group)
        cluster_dfs = get_dataframes_per_cluster(cluster_treatment, dataframes)
        for cluster,cluster_treatments in cluster_treatment.items():
            X = cluster_dfs[cluster][1].drop(columns=['treatment_id'])
            Y = cluster_dfs[cluster][2]
            X_train,X_test_s,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
            Y_train = Y_train.drop(columns=['treatment_id'])
            test_sets = (X_train, X_test_s, Y_test, cluster_dfs[cluster][3]) 
            for algorithm, model_data in fitted_models[cluster].items():
                print(algorithm)
                model = model_data[1]
                if algorithm in ['RF','DT']:
                    model = GridSearchCV(model,error_score=0.0, cv=5).best_estimator_
                else:
                    model = GridSearchCV(model, cv=5).best_estimator_
                model = model.fit()
                test_sets[2]['Y_pred'] = model.predict(X_test_s)
                store_metrics_per_cluster(app, feature_group[0], cluster_treatments, model_data[0], test_sets, run_id)
    
def do_step_7b(app):
    folder="results/clustering/pkl_objects"
    start_logging(folder)
    run_type = 'definite'
    run_id = get_run_id(app, 'Predictions per treatment Grid Search', run_type, '7b', 'vfc')
    print('Getting dataframe')
    dataframes = get_dataframe_per_treatment(app)
    print('Getting feature groups')
    feature_groups = get_feature_groups_for_prediction(app, run_type)
    print("Step 7b: Getting predictions")
    get_predictions(app, folder, feature_groups, dataframes, run_id)
    complete_run(app, run_id)
