import pandas as pd
import sys
import logging

from sklearn.metrics import adjusted_rand_score, rand_score, mutual_info_score, adjusted_mutual_info_score,\
                            fowlkes_mallows_score, homogeneity_completeness_v_measure

from sqlalchemy import and_

from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run
from database.models.hft_tables import HFT_MODEL_T, HFT_MODEL_PARAMETERS_T, HFT_METRICS_T, HFT_TREATMENT_T\
                                     , HFT_DATA_CHARACTERISTIC_T, HFT_DATASET_CHARACTERISTIC_T\
                                     , CLUSTER_FEATURES_GROUP_T, CLUSTER_FEATURES_T, CLUSTER_GENERATED_T\
                                     , CLUSTER_METRICS_T, CLUSTER_TREATMENT_T
from steps.step_generic_code.general_variables.general_variables_dataset_clustering import CLASSES_VALUES, CLUSTERING_MODELS

def getCombination(arr, n, r):
    data = [0]*r
    result = []
    return combinationUtil(arr, data, 0, n - 1, 0, r, result)

def combinationUtil(arr, data, start, end, index, r, result):
    if (index == r):
        result.append(data.copy())
        return

    i = start; 
    while(i <= end and end - i + 1 >= r - index):
        data[index] = arr[i]
        combinationUtil(arr, data, i + 1, end, index + 1, r, result)
        i += 1
    return result

def get_data_characteristics(app):
    characteristics = {}
    data_characteristics = app.session.query(HFT_DATA_CHARACTERISTIC_T.id, HFT_DATA_CHARACTERISTIC_T.characteristic_name).all()
    for record in data_characteristics:
        characteristics[record[1]]=record[0]
    return characteristics
 
def get_characteristics(app, run_id):
    result = []
    characteristic_names = get_data_characteristics(app)
    treatment_ids = app.session.query(HFT_DATASET_CHARACTERISTIC_T.hft_treatment_id)\
                               .filter(HFT_DATASET_CHARACTERISTIC_T.hft_run_id==run_id).distinct().all()
    for tp1 in treatment_ids:
        treatment_id = tp1[0]
        characteristics = app.session.query(HFT_DATASET_CHARACTERISTIC_T).\
            filter(and_(HFT_DATASET_CHARACTERISTIC_T.hft_treatment_id==treatment_id,
                        HFT_DATASET_CHARACTERISTIC_T.hft_run_id==run_id)).all()
        if characteristics:
            row = {}
            for characteristic in characteristics:
                row[characteristic.characteristic_id] = characteristic.value
            complete=True
            list_row=[treatment_id]
            for id in characteristic_names.values():
                if id in row:
                    list_row.append(row[id])
                else:
                    logging.info(id)
                    complete=False
            if complete:
                result.append(list_row)
            else:
                logging.info('Incomplete characteristics for treatment_id: ' + str(treatment_id))
    return characteristic_names, result

def get_model_id_best_model(metrics):
    max_accuracy=0
    max_f1=0
    model_id=-1
    for metric in metrics:
        if ((metric.accuracy>max_accuracy) or ((metric.accuracy==max_accuracy) and (metric.f1_score>max_f1))):
            max_accuracy=metric.accuracy
            max_f1=metric.f1_score
            model_id=metric.hft_model_id
    return model_id

def get_model_values(app,characteristics, run_ids):
    result=[]
    for characteristic in characteristics:
        treatment_id = characteristic[0]
        metrics = app.session.query(HFT_METRICS_T).filter(and_(HFT_METRICS_T.hft_treatment_id==treatment_id,
                                                               HFT_METRICS_T.hft_run_id.in_(run_ids))).all()
        if metrics:
            result_row = characteristic
            model_id = get_model_id_best_model(metrics)
            result_row.append(model_id)
            model = app.session.query(HFT_MODEL_T.algorithm).filter(HFT_MODEL_T.id==model_id).first()
            result_row.append(CLASSES_VALUES[model.algorithm])
            result.append(result_row)
    return result

def get_parameter_values(app, characteristics, run_ids):
    results = []
    for characteristic in characteristics:
        treatment_id = characteristic[0]
        rf_parameter_values = app.session.query(HFT_MODEL_PARAMETERS_T)\
                                         .join(HFT_MODEL_T, HFT_MODEL_T.id==HFT_MODEL_PARAMETERS_T.hft_model_id)\
                                         .join(HFT_METRICS_T, HFT_METRICS_T.hft_model_id==HFT_MODEL_T.id)\
                                         .filter(and_(HFT_METRICS_T.hft_run_id.in_(run_ids), HFT_MODEL_T.algorithm=='RF',
                                                      HFT_MODEL_PARAMETERS_T.hft_parameters_t_id==18,
                                                      HFT_MODEL_PARAMETERS_T.treatment_id==treatment_id)).all()
        if rf_parameter_values:
            result_row = characteristic
            result_row.append(rf_parameter_values[0].value)
            results.append(result_row)
        else:
            logging.info('No value found for: ' + str(treatment_id))
    return results

def add_patient_group(app, characteristics):
    patient_group_query = app.session.query(HFT_TREATMENT_T.treatment_id, HFT_TREATMENT_T.research_group).all()
    patient_group = pd.DataFrame([row for row in patient_group_query], columns=['treatment_id','research_group'])
    return pd.merge(characteristics, patient_group, left_on=['treatment_id'], right_on=['treatment_id'], how='left')

def save_cluster_info(app, combination, algorithm, run_id, nr_of_features, characteristic_names):
    feature_group = CLUSTER_FEATURES_GROUP_T(nr_of_features = nr_of_features, 
                                                hft_run_id = run_id,
                                                algorithm = algorithm)
    app.session.add(feature_group)
    app.session.commit()
    for parameter in combination:
        feature = CLUSTER_FEATURES_T(hft_feature_group_id = feature_group.id, 
                                    hft_parameters_t_id = characteristic_names[parameter])
        app.session.add(feature)
    app.session.commit()
    return feature_group.id

def save_generated_clusters(app, characteristics, feature_group_id):
    for result, clustering in characteristics.groupby(['cluster_result','n_estimators']):
        cluster_result = CLUSTER_GENERATED_T(cluster_group = result[0],
                                            original_group = result[1],
                                            cluster_size = clustering.shape[0],
                                            hft_feature_group = feature_group_id)
        app.session.add(cluster_result)
    app.session.commit()

def save_cluster_treatment(app, characteristics, feature_group_id):
    def save_cluster_appointment(row):
        cluster_treatment = CLUSTER_TREATMENT_T(hft_treatment_id = row['treatment_id'],
                                                cluster_generated = row['id'])
        app.session.add(cluster_treatment)

    clusters_generated_query = app.session.query(CLUSTER_GENERATED_T.id, CLUSTER_GENERATED_T.cluster_group, 
                                                    CLUSTER_GENERATED_T.original_group)\
                                            .filter(CLUSTER_GENERATED_T.hft_feature_group==feature_group_id).all()
    clusters_generated = pd.DataFrame([row for row in clusters_generated_query], columns=['id', 'cluster_group', 'original_group'])
    cluster_treatment = pd.merge(characteristics, clusters_generated, left_on=['cluster_result','n_estimators'], right_on=['cluster_group', 'original_group'], how='left')
    cluster_treatment.apply(save_cluster_appointment, axis=1)
    app.session.commit()

def clustering(app, characteristics, characteristic_names, run_id, nr_clusters, nr_features_start):
    names = [name for name in characteristic_names.keys()]
    total_features = len(names)
    for algorithm, model in CLUSTERING_MODELS[nr_clusters]: 
        for nr_of_features in range(nr_features_start, 10):
            print(algorithm, nr_of_features)
            combinations = getCombination(names, total_features, nr_of_features)
            for combination in combinations:
                feature_group_id = save_cluster_info(app, combination, algorithm, run_id, nr_of_features, characteristic_names)
                X = characteristics[characteristics.columns.intersection(combination)]
                fitted_model = model.fit(X)
                characteristics['cluster_result'] = fitted_model.labels_
                save_generated_clusters(app, characteristics, feature_group_id)
                save_cluster_treatment(app, characteristics, feature_group_id)

def get_metrics(app, run_id):
    def add_research_cluster(row):
        if row['research_group'] in (1,2):
            row['original_research_group'] = row['original_group'] + "_vfc"
        else:
            row['original_research_group'] = row['original_group'] + "_ns"
        return row

    print('Saving metrics')
    groups = app.session.query(CLUSTER_FEATURES_GROUP_T.id).filter(CLUSTER_FEATURES_GROUP_T.hft_run_id==run_id).all()
    for feature_group_id in groups:
        clusters = app.session.query(CLUSTER_GENERATED_T.cluster_group, CLUSTER_GENERATED_T.original_group, 
                                     CLUSTER_TREATMENT_T.hft_treatment_id, HFT_TREATMENT_T.research_group)\
                              .join(CLUSTER_TREATMENT_T, CLUSTER_TREATMENT_T.cluster_generated==CLUSTER_GENERATED_T.id)\
                              .join(HFT_TREATMENT_T, HFT_TREATMENT_T.treatment_id==CLUSTER_TREATMENT_T.hft_treatment_id)\
                              .filter(CLUSTER_GENERATED_T.hft_feature_group==feature_group_id[0])
        cluster_df = pd.DataFrame([row for row in clusters], columns=['cluster_group', 'original_group', 'treatment_id', 'research_group'])
        cluster_df['original_research_group'] = cluster_df['original_group']
        cluster_df = cluster_df.apply(add_research_cluster, axis=1)
        cluster_rand_score = rand_score(cluster_df['cluster_group'],cluster_df['original_research_group'])
        cluster_adjusted_rand_score = adjusted_rand_score(cluster_df['cluster_group'],cluster_df['original_research_group'])
        cluster_MI = mutual_info_score(cluster_df['cluster_group'],cluster_df['original_research_group'])
        cluster_NMI = adjusted_mutual_info_score(cluster_df['cluster_group'],cluster_df['original_research_group'])
        cluster_HCV = homogeneity_completeness_v_measure(cluster_df['cluster_group'],cluster_df['original_research_group'])
        cluster_FMS = fowlkes_mallows_score(cluster_df['cluster_group'],cluster_df['original_research_group'])
        cluster_metric = CLUSTER_METRICS_T(hft_feature_group_id = feature_group_id[0],
                                           rand_index = cluster_rand_score,
                                           adjusted_rand_index = cluster_adjusted_rand_score,
                                           fowlkes_mallows_score = cluster_FMS,
                                           homogeneity = cluster_HCV[0],
                                           completeness = cluster_HCV[1],
                                           v_measure = cluster_HCV[2],
                                           mutual_info = cluster_MI,
                                           adjusted_mutual_info = cluster_NMI)
        app.session.add(cluster_metric)
    app.session.commit()

def do_clustering(app, df_characteristics, names, run_id, nr_features_start=2, nr_clusters='5'):
    type = 'vfc'
    if len(sys.argv)>=6:
        type = sys.argv[5]
    df_used = df_characteristics
    if type=='vfc':
        df_used = df_characteristics.loc[df_characteristics['research_group'].isin([1,2])]
    elif type=='ns':
        df_used = df_characteristics.loc[df_characteristics['research_group'].isin([3,4])]
    df_used = df_used.reset_index()
    if len(sys.argv)==7:
        nr_clusters = sys.argv[6]
    print('Starting clustering for ' + type + ' with ' + nr_clusters + ' clusters')
    clustering(app, df_used, names, run_id, nr_clusters, nr_features_start)

def do_step_6(app):
    folder="results/clustering"
    characteristics_run_id = 139
    fitting_run_ids = [14, 16]
    start_logging(folder)
    run_id = get_run_id(app, 'Clustering on all characterisitics', 'test', 6, 'both')
    names, characteristics = get_characteristics(app, characteristics_run_id)
    characteristics = get_model_values(app, characteristics, fitting_run_ids)
    characteristics = get_parameter_values(app, characteristics, fitting_run_ids)
    colummns = ['treatment_id'] + [name for name in names.keys()] + ['model_id', 'class', 'n_estimators']
    df_characteristics = pd.DataFrame(characteristics, columns=colummns)
    df_characteristics = add_patient_group(app, df_characteristics)
    do_clustering(app, df_characteristics, names, run_id)
    get_metrics(app,run_id)
    complete_run(app, run_id)
