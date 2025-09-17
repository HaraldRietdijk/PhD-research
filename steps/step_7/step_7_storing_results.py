import pickle, os
import pandas as pd

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from database.models.hft_tables import HFT_MODEL_T, HFT_MODEL_PARAMETERS_T, HFT_METRICS_T,\
                                       FEATURE_GROUP_CLUSTER_METRICS, FEATURE_GROUP_CLUSTER_TIMES

from steps.step_generic_code.general_functions import pickle_destination
from steps.step_generic_code.general_variables.general_variables_dataset_clustering import ALGORITHM_PARAMETERS
from steps.step_3.step_3_generic_storing_results import find_or_create_parameter_id, convert_list

def insert_hyperparameters_into_database(app, model_id, treatment_id, algorithm_name, model):
    for parameter_name in ALGORITHM_PARAMETERS[algorithm_name]:
        parameter_id = find_or_create_parameter_id(app,algorithm_name,parameter_name)
        value=getattr(model.best_estimator_ , parameter_name)
        if type(value) is list:
            value = convert_list(value)
        app.session.add(HFT_MODEL_PARAMETERS_T(hft_model_id = model_id, 
                                               treatment_id = treatment_id,
                                               hft_parameters_t_id = parameter_id,
                                               value = value))
    app.session.commit()

def save_model_to_disk(dest, cluster_id, classifier_name, fitted_model):
    pickle_model = cluster_id + '_' + classifier_name + '_' + 'model.pkl'
    pickle.dump(fitted_model.best_estimator_,open(os.path.join(dest, pickle_model),'wb'),protocol=4)
    return pickle_model

def save_model_and_pickle(app, folder, cluster_id, classifier_name, fitting_tuple):
    sub_folder='pkl_objects'
    dest = pickle_destination(folder,sub_folder)
    fitted_model = fitting_tuple[0]
    pickle_model = save_model_to_disk(dest, cluster_id, classifier_name, fitted_model)
    model=HFT_MODEL_T(name = pickle_model, 
                      algorithm = classifier_name, 
                      destination = dest, 
                      random_seed = 10)
    app.session.add(model)
    app.session.commit()
    model_id = model.id
    insert_hyperparameters_into_database(app, model_id, 0, classifier_name, fitted_model)    
    return model.id

def store_fitting_times(app, fitting_times_all_clusters, feature_group_id, run_id):
    for cluster_name, times_per_model in fitting_times_all_clusters.items():
        for classifier, fitting_results in times_per_model.items():
            app.session.add(FEATURE_GROUP_CLUSTER_TIMES(hft_feature_group = feature_group_id,
                                                        cluster_group = cluster_name,
                                                        hft_run_id = run_id,
                                                        algorithm = classifier,
                                                        fitting_time_sec = fitting_results[1]))
    app.session.commit()

def store_metrics(app, feature_group_id, cluster_name, model_id, Y, Y_pred, run_id):
    df_confusion_matrix = pd.DataFrame(confusion_matrix(Y, Y_pred))
    TN = df_confusion_matrix.iloc[0,0]
    FP = df_confusion_matrix.iloc[0,1]
    FN = df_confusion_matrix.iloc[1,0]
    TP = df_confusion_matrix.iloc[1,1]
    F1 = f1_score(Y,Y_pred, average='macro')
    ACC = accuracy_score(Y, Y_pred)
    OBS = len(Y)
    app.session.add(FEATURE_GROUP_CLUSTER_METRICS(hft_feature_group = feature_group_id, 
                                                  cluster_group = cluster_name,
                                                  hft_model_id = model_id,
                                                  f1_score = F1, 
                                                  true_negative = TN, 
                                                  true_positive = TP,
                                                  false_negative = FN, 
                                                  false_positive = FP, 
                                                  accuracy = ACC,
                                                  number_of_observations = OBS, 
                                                  hft_run_id = run_id))
    app.session.commit()

def store_metrics_per_treatment(app, feature_group, treatment, model_id, dataframes, Y_pred, run_id):
    Y = dataframes[2]
    df_confusion_matrix = pd.DataFrame(confusion_matrix(Y,Y_pred))
    THRESHOLD = dataframes[3]
    TN = df_confusion_matrix.iloc[0,0]
    FP = df_confusion_matrix.iloc[0,1]
    FN = df_confusion_matrix.iloc[1,0]
    TP = df_confusion_matrix.iloc[1,1]
    F1 = f1_score(Y,Y_pred, average='macro')
    ACC = accuracy_score(Y, Y_pred)
    OBS = len(Y)
    WEEKDAY = feature_group
    app.session.add(HFT_METRICS_T(f1_score = F1, 
                                  true_negative = TN, 
                                  true_positive = TP,
                                  false_negative = FN, 
                                  false_positive = FP, 
                                  accuracy = ACC,
                                  hft_model_id = model_id, 
                                  hft_treatment_id = treatment, 
                                  threshold = THRESHOLD, 
                                  number_of_observations = OBS, 
                                  weekday = WEEKDAY,
                                  hft_run_id = run_id))
    app.session.commit()

def store_metrics_per_cluster(app, feature_group, cluster_treatments, model_id, dataframes, run_id):
    all_Y = dataframes[2]
    for treatment in cluster_treatments:
        Y = all_Y.loc[all_Y['treatment_id']==treatment]['dailysteps_cat']
        Y_pred = all_Y.loc[all_Y['treatment_id']==treatment]['Y_pred']
        df_confusion_matrix = pd.DataFrame(confusion_matrix(Y,Y_pred))
        THRESHOLD = dataframes[3]
        TN = df_confusion_matrix.iloc[0,0]
        FP = df_confusion_matrix.iloc[0,1]
        FN = df_confusion_matrix.iloc[1,0]
        TP = df_confusion_matrix.iloc[1,1]
        F1 = f1_score(Y,Y_pred, average='macro')
        ACC = accuracy_score(Y, Y_pred)
        OBS = len(Y)
        WEEKDAY = feature_group
        app.session.add(HFT_METRICS_T(f1_score = F1, 
                                    true_negative = TN, 
                                    true_positive = TP,
                                    false_negative = FN, 
                                    false_positive = FP, 
                                    accuracy = ACC,
                                    hft_model_id = model_id, 
                                    hft_treatment_id = treatment, 
                                    threshold = THRESHOLD, 
                                    number_of_observations = OBS, 
                                    weekday = WEEKDAY,
                                    hft_run_id = run_id))
        app.session.commit()
