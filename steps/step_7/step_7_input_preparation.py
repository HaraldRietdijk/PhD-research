import pandas as pd
import numpy as np
import pickle, os
from sqlalchemy import and_, func

from database.models.hft_tables import HFT_RUN_T, CLUSTERING_SELECTED_T, CLUSTER_GENERATED_T,\
                                       CLUSTER_TREATMENT_T, CLUSTER_RUN_INFO_T, CLUSTER_FEATURES_GROUP_T, CLUSTER_METRICS_T,\
                                       FEATURE_GROUP_CLUSTER_METRICS, HFT_MODEL_T
from steps.step_generic_code.dataframes_step_data import get_dataframe, get_cat, get_threshold,\
                                                         get_dataframe_ns, get_threshold_ns, get_trainsets, get_full_sets

def get_dataframes(app):
    df_dataframes = {}
    treatments, df_dataframe = get_dataframe(app)
    df_dataframe = get_threshold(df_dataframe)
    df_dataframe['dailysteps_cat'] = get_cat(df_dataframe, df_dataframe)
    df_dataframes['vfc']=df_dataframe
    treatments, df_dataframe = get_dataframe_ns(app)
    df_dataframe = get_threshold_ns(app, 'LIN_W', df_dataframe)
    df_dataframe['dailysteps_cat'] = get_cat(df_dataframe, df_dataframe)
    df_dataframes['ns'] = df_dataframe
    df_dataframes['both'] = pd.concat([df_dataframes['vfc'],df_dataframes['ns']])
    return df_dataframes

def get_run_ids(app):
    run_ids_query = app.session.query(HFT_RUN_T.run_id, HFT_RUN_T.data_set, CLUSTER_RUN_INFO_T.nr_of_clusters)\
                               .join(CLUSTER_RUN_INFO_T, CLUSTER_RUN_INFO_T.hft_run_id == HFT_RUN_T.run_id)\
                               .filter(and_(HFT_RUN_T.run_step==6, HFT_RUN_T.run_completed==1, 
                                            HFT_RUN_T.run_type=='definite', CLUSTER_RUN_INFO_T.selected==1)).all()
    return [{'run_id': row[0], 'dataset': row[1], 'nr_of_clusters': row[2]} for row in run_ids_query]

def select_feature_groups(app, run_ids):
    selected = pd.DataFrame()
    first = True
    for run_ids_row in run_ids:
        feature_groups_query = app.session.query(CLUSTER_FEATURES_GROUP_T.id, CLUSTER_FEATURES_GROUP_T.algorithm,
                                                 CLUSTER_FEATURES_GROUP_T.nr_of_features, CLUSTER_METRICS_T.rand_index)\
                                  .join(CLUSTER_METRICS_T, CLUSTER_METRICS_T.hft_feature_group_id==CLUSTER_FEATURES_GROUP_T.id)\
                                  .filter(CLUSTER_FEATURES_GROUP_T.hft_run_id==run_ids_row['run_id'])\
                                  .all()
        feature_groups = pd.DataFrame([row for row in feature_groups_query], 
                                      columns=['id','algorithm','nr_of_features','rand_index'])
        feature_groups['run_id'] = run_ids_row['run_id']
        feature_groups['dataset'] = run_ids_row['dataset']
        feature_groups['nr_of_clusters'] = run_ids_row['nr_of_clusters']
        feature_groups['maximum'] = np.where(feature_groups.rand_index.eq(feature_groups.groupby('algorithm')['rand_index'].transform('max')), 'yes', 'no')
        to_be_added = feature_groups.loc[feature_groups['maximum']=='yes']
        if first:
            first = False
            selected = to_be_added
        else:
            selected = pd.concat([selected, to_be_added])
    return selected

def select_feature_groups_min(app, run_ids):
    selected = pd.DataFrame()
    first = True
    for run_ids_row in run_ids:
        feature_groups_query = app.session.query(CLUSTER_FEATURES_GROUP_T.id, CLUSTER_FEATURES_GROUP_T.algorithm,
                                                 CLUSTER_FEATURES_GROUP_T.nr_of_features, CLUSTER_METRICS_T.rand_index,
                                                 func.count(CLUSTER_FEATURES_GROUP_T.id))\
                                  .join(CLUSTER_METRICS_T, CLUSTER_METRICS_T.hft_feature_group_id==CLUSTER_FEATURES_GROUP_T.id)\
                                  .join(CLUSTER_GENERATED_T, CLUSTER_GENERATED_T.hft_feature_group==CLUSTER_FEATURES_GROUP_T.id)\
                                  .filter(and_(CLUSTER_FEATURES_GROUP_T.hft_run_id==run_ids_row['run_id'],
                                               CLUSTER_FEATURES_GROUP_T.algorithm.startswith('SPECCL')))\
                                  .group_by(CLUSTER_FEATURES_GROUP_T.id, CLUSTER_FEATURES_GROUP_T.algorithm,
                                                 CLUSTER_FEATURES_GROUP_T.nr_of_features, CLUSTER_METRICS_T.rand_index)\
                                  .having(func.count(CLUSTER_FEATURES_GROUP_T.id) > 7).all()
        feature_groups = pd.DataFrame([row for row in feature_groups_query], 
                                      columns=['id','algorithm','nr_of_features','rand_index', 'count'])
        feature_groups=feature_groups.drop(['count'], axis=1)
        feature_groups['run_id'] = run_ids_row['run_id']
        feature_groups['dataset'] = run_ids_row['dataset']
        feature_groups['nr_of_clusters'] = run_ids_row['nr_of_clusters']
        feature_groups['minimum'] = np.where(feature_groups.rand_index.eq(feature_groups.groupby('algorithm')['rand_index'].transform('min')), 'yes', 'no')
        all_minimums = feature_groups.loc[feature_groups['minimum']=='yes']
        to_be_added = all_minimums.head(2)
        if first:
            first = False
            selected = to_be_added
        else:
            selected = pd.concat([selected, to_be_added])
    return selected

def get_feature_groups(app, run_id):
# Getting the feature groups
# First get the id's of the relevant runs, with their dataset and number of clusters info
# Get the feature groups per run and findd the feature group with the maximum Rand_index per 
# group for each algorithm that was used in the run.
# Finally save the selection to the database.
    def store_selection(row):
        clustering_selected = CLUSTERING_SELECTED_T(run_id_origin = row['run_id'],
                                                    hft_feature_group = row['id'],
                                                    hft_run_id = run_id)
        app.session.add(clustering_selected)

    # run_ids = get_run_ids(app)
    run_ids = [{'run_id': 156, 'dataset': 'vfc', 'nr_of_clusters': 4},{'run_id': 158, 'dataset': 'vfc', 'nr_of_clusters': 7}]
    # feature_groups = select_feature_groups(app, run_ids)
    feature_groups = select_feature_groups_min(app, run_ids)
    print('Save selection')
    feature_groups.apply(store_selection, axis=1) 
    app.session.commit()
    return feature_groups

def get_clusters(app, feature_group_id):
    clusters = app.session.query(CLUSTER_GENERATED_T.cluster_group, CLUSTER_TREATMENT_T.hft_treatment_id)\
                          .join(CLUSTER_TREATMENT_T, CLUSTER_TREATMENT_T.cluster_generated==CLUSTER_GENERATED_T.id)\
                          .filter(CLUSTER_GENERATED_T.hft_feature_group==feature_group_id).all()
    return pd.DataFrame([row for row in clusters], columns=['cluster_group', 'treatment_id'])


def get_dataframe_per_cluster(app, feature_group, df_dataframes):
    cluster_dfs = {}
    clusters = get_clusters(app, feature_group['id'])
    df_used = df_dataframes[feature_group['dataset']]
    df_used = pd.merge(df_used, clusters, left_on='treatment_id', right_on='treatment_id', how='inner')
    for cluster_name, group in df_used.groupby('cluster_group'):
        cluster_dfs[str(cluster_name)] = group
    return cluster_dfs

def split_dataframes(cluster_dfs, variable_threshold):
    cluster_sets = {}
    for cluster, dataframe in cluster_dfs.items():
        sets = {}
        sets['X_train_s'], _ , sets['X_train'], sets['Y_train'] = get_trainsets(dataframe, 0.3, variable_threshold)
        cluster_sets[cluster] = sets
    return cluster_sets

def get_dataframe_per_treatment(app):
    dataframes = {}
    df_treatments, df_dataframe = get_dataframe(app)
    df_dataframe = get_threshold(df_dataframe)
    df_dataframe['dailysteps_cat'] = get_cat(df_dataframe, df_dataframe)
    threshold = df_dataframe['threshold'].mean()
    X, X_scaled, Y = get_full_sets(df_dataframe, variable_threshold=False, include_treatment=True)
    for treatment_id in df_treatments['treatment_id']:
        X_treatment = X.loc[df_dataframe['treatment_id']==treatment_id]
        # X_treatment = X_treatment.drop(['treatment_id'], axis=1)
        X_scaled_treatment = X_scaled.loc[df_dataframe['treatment_id']==treatment_id]
        # X_scaled_treatment = X_scaled_treatment.drop(['treatment_id'], axis=1)
        Y_treatment = Y.loc[df_dataframe['treatment_id']==treatment_id]
        # Y_treatment = Y_treatment.drop(['treatment_id'], axis=1)
        dataframes[str(treatment_id)] = (X_treatment, X_scaled_treatment, Y_treatment, threshold)
    return dataframes

def get_feature_groups_for_prediction(app, run_type):
    run_ids_select = app.session.query(HFT_RUN_T.run_id).filter(and_(HFT_RUN_T.run_step=='7', HFT_RUN_T.run_type==run_type, 
                                                              HFT_RUN_T.run_completed==1, HFT_RUN_T.data_set=='vfc')).all()
    run_ids = [row for row in run_ids_select]
    feature_groups = []
    for run_id in run_ids:
        feature_groups_select = app.session.query(CLUSTERING_SELECTED_T.hft_feature_group).filter(CLUSTERING_SELECTED_T.hft_run_id==run_id[0]).all()
        feature_groups.extend([(row[0], run_id[0]) for row in feature_groups_select])
    return feature_groups

def get_cluster_per_treatment(app, feature_group):
    cluster_select = app.session.query(CLUSTER_GENERATED_T.cluster_group, CLUSTER_TREATMENT_T.hft_treatment_id)\
                                .join(CLUSTER_TREATMENT_T, CLUSTER_GENERATED_T.id==CLUSTER_TREATMENT_T.cluster_generated)\
                                .filter(CLUSTER_GENERATED_T.hft_feature_group==feature_group[0]).all()
    return { str(row[1]) : str(row[0]) for row in cluster_select}

def get_treatments_per_cluster(app, feature_group):
    treatments_per_cluster = {}
    cluster_select = app.session.query(CLUSTER_GENERATED_T.cluster_group, CLUSTER_TREATMENT_T.hft_treatment_id)\
                                .join(CLUSTER_TREATMENT_T, CLUSTER_GENERATED_T.id==CLUSTER_TREATMENT_T.cluster_generated)\
                                .filter(CLUSTER_GENERATED_T.hft_feature_group==feature_group[0]).all()
    for row in cluster_select:
        if not (str(row[0]) in treatments_per_cluster.keys()):
            treatments_per_cluster[str(row[0])]=[row[1]]
        else:
            treatments_per_cluster[str(row[0])].append(row[1])
    return treatments_per_cluster

def get_dataframes_per_cluster(cluster_treatment, dataframes):
    cluster_dfs = {}
    for cluster, treatments in cluster_treatment.items():
        X = []
        X_scaled = []
        Y = []
        for treatment_id in treatments:
            treatment = str(treatment_id)
            X.append(dataframes[treatment][0])
            X_scaled.append(dataframes[treatment][1])
            Y.append(dataframes[treatment][2])
            treshold = dataframes[treatment][3]
        cluster_dfs[cluster] = (pd.concat(X), pd.concat(X_scaled), pd.concat(Y), treshold)
    return cluster_dfs

def get_models(app, feature_group):
    models = {}
    models_select = app.session.query(FEATURE_GROUP_CLUSTER_METRICS.cluster_group, FEATURE_GROUP_CLUSTER_METRICS.hft_model_id,
                                      HFT_MODEL_T.algorithm, HFT_MODEL_T.name)\
                               .join(HFT_MODEL_T, HFT_MODEL_T.id==FEATURE_GROUP_CLUSTER_METRICS.hft_model_id)\
                               .filter(and_(FEATURE_GROUP_CLUSTER_METRICS.hft_feature_group==feature_group[0], 
                                            FEATURE_GROUP_CLUSTER_METRICS.hft_run_id==feature_group[1])).all()
    for row in models_select:
        if not (str(row[0]) in models.keys()):
            models[str(row[0])]={}
        models[str(row[0])][row[2]]=(row[1], row[3])
    return models

def add_pickled_models(folder, fitted_models):
    updated_models = {}
    for key, algorithms in fitted_models.items():
        updated_models[key]={}
        for algorithm, value in algorithms.items():
            updated_models[key][algorithm] = (value[0], pickle.load(open(os.path.join(folder,value[1]), 'rb')))
    return updated_models

