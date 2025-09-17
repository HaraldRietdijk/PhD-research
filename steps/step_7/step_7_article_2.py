
import pandas as pd
import numpy as np
from sqlalchemy import and_, or_, not_, func
import os 
import sys
import pickle
import logging
import time

from steps.step_generic_code.dataframes_step_data import daily_steps_cat_f
from database.models.hft_tables import CLUSTER_TREATMENT_T, CLUSTER_FEATURES_GROUP_T, CLUSTERING_SELECTED_T, HFT_TREATMENT_T
from database.models.hft_views import SumSteps

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, PolynomialFeatures, RobustScaler, StandardScaler

from database.models.hft_tables import HFT_ALGORITHM_T, HFT_MODEL_T, HFT_MODEL_PARAMETERS_T, HFT_METRICS_T, HFT_PARAMETERS_T, HFT_FITTING_TIME_T

from steps.step_generic_code.general_variables.general_variables_cluster import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS
from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run
cluster_dfs = {}
threshold_dfs = {}
x = {}
y = {}
x_s ={}
x_train = {}
y_train = {}
x_train_s = {}
y_train_s = {}
y_pred = {}
models = {}
classifiers =  { name: [classifier, normalized] for name, classifier, normalized in CLASSIFIERS}

def save_model_to_disk(dest,cluster_code):
    pickle_model=cluster_code+'_'+'model.pkl' 
    pickle.dump(models[cluster_code].best_estimator_,open(os.path.join(dest,pickle_model),'wb'),protocol=4)

def pickle_destination(root,leaf):
    dest = os.path.join(root,leaf) 
    if not os.path.exists(dest): 
        os.makedirs(dest) 
    return dest    

def get_treatments_per_cluster(app):
    clusters = app.session.query(CLUSTER_FEATURES_GROUP_T).all()
    treatments_per_cluster = {}
    for cluster in clusters:
        cluster_treatment = app.session.query(CLUSTER_TREATMENT_T).filter(CLUSTER_TREATMENT_T.cluster_generated==cluster.id).all()
        treatments_per_cluster[cluster.id] = [row.hft_treatment_id for row in cluster_treatment]
    return treatments_per_cluster

def load_dataframe_and_thresholds(app, treatments_per_cluster):
    global cluster_dfs, threshold_dfs
    print("Step 7: Getting dataframes")
    for cluster_code, treatment_ids in treatments_per_cluster.items():
        dataframe_result = app.session.query(SumSteps.id, SumSteps.treatment_id, SumSteps.year, SumSteps.week, SumSteps.weekday,
                                    SumSteps.hour, SumSteps.sum_steps, SumSteps.sum_steps_hour, SumSteps.daily_steps)\
                            .filter(and_(SumSteps.treatment_id.in_(treatment_ids),
                                        SumSteps.hour.in_([7,8,9,10,11,12,13,14,15,16,17,18]),
                                        not_(SumSteps.weekday.in_([5,6])),
                                        SumSteps.year==2015,
                                        or_(and_(SumSteps.research_group==2,SumSteps.week>15),
                                            and_(SumSteps.research_group==1,SumSteps.week>4))))\
                            .order_by(SumSteps.year,SumSteps.week,SumSteps.weekday,SumSteps.hour)
        cluster_dfs[cluster_code]=pd.DataFrame([row for row in dataframe_result],
                                                    columns=['id','treatment_id','year','week','weekday','hour','sum_steps','sum_steps_hour','daily_steps'])
        threshold_result = app.session.query(func.avg(SumSteps.sum_steps_hour))\
                            .filter(and_(SumSteps.treatment_id.in_(treatment_ids),
                                        SumSteps.hour==18,
                                        not_(SumSteps.weekday.in_([5,6])),
                                        SumSteps.year==2015,
                                        or_(and_(SumSteps.research_group==2,SumSteps.week>15),
                                            and_(SumSteps.research_group==1,SumSteps.week>4))))
        threshold_dfs[cluster_code] = pd.DataFrame([row for row in threshold_result],
                                                                columns=['avg_daily_steps'])
        threshold_dfs[cluster_code]['cluster_id'] = cluster_code
        x=cluster_dfs[cluster_code]['daily_steps']
        y=threshold_dfs[cluster_code]['avg_daily_steps']
        cluster_dfs[cluster_code]['dailysteps_cat']=np.vectorize(daily_steps_cat_f)(x, y)
    print('\n')

def split_dataframes(treatments_per_cluster, random_seed=10):
    global cluster_dfs, x_train, y_train, x_train_s, y_train_s
    print("Step 7: Splitting dataframes")
    for cluster_code in treatments_per_cluster.keys():
        X = cluster_dfs[cluster_code].iloc[:, 5:8].values
        y = cluster_dfs[cluster_code].iloc[:, 9].values
        #normalize for NN,SVC,SGD and KNN
        min_max_scaler = MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(X)
        X_s = pd.DataFrame(np_scaled)
        x_train[cluster_code],x_test,y_train[cluster_code],y_test= train_test_split(X,y, test_size=0.3, random_state=random_seed)    
        x_train_s[cluster_code],x_test_s,y_train_s[cluster_code],y_test_s= train_test_split(X_s,y, test_size=0.3, random_state=random_seed)    

def fit_model (cluster_code,use_algorithm,X_train,y_train):
    global models
    parameters=FITTING_PARAMETERS[cluster_code]
    log_text="Cluster {c}".format(c=cluster_code)
    logging.info(log_text)
    start = time.time()
    models[cluster_code] = GridSearchCV(use_algorithm, parameters,error_score=0.0, cv=5).fit(X_train, y_train)
    end = time.time()
    return (end-start)

def fit_models():
    global cluster_dfs
    print("Step 7: Fitting models")
    fitting_times_all_clusters=dict()
    for cluster_code in cluster_dfs.keys():
        if classifiers[cluster_code][1]:
            fitting_time=fit_model(cluster_code, classifiers[cluster_code][0],x_train_s[cluster_code],y_train_s[cluster_code])
        else:
            fitting_time=fit_model(cluster_code, classifiers[cluster_code][0],x_train[cluster_code],y_train[cluster_code])
        fitting_times_all_clusters[cluster_code]=fitting_time
        print('\n')
    return fitting_times_all_clusters

def make_predictions():
    global cluster_dfs, models, y_pred, x, y, x_s
    print("Step 7: making predictions")
    for cluster_code, cluster_df in cluster_dfs.items():
        x[cluster_code] = cluster_df.iloc[:, 5:8].values
        y[cluster_code] = cluster_df.iloc[:,9].values
        if classifiers[cluster_code][1]:
            min_max_scaler = MinMaxScaler()
            np_scaled = min_max_scaler.fit_transform(x[cluster_code])
            x_s[cluster_code] = pd.DataFrame(np_scaled)
            y_pred[cluster_code] = models[cluster_code].predict(x_s[cluster_code])
        else:
            y_pred[cluster_code] = models[cluster_code].predict(x[cluster_code])


def check_algorithm(app, algorithm_name):
    algorithm = app.session.query(HFT_ALGORITHM_T).filter(HFT_ALGORITHM_T.name==algorithm_name).first()
    if not algorithm:
        app.session.add(HFT_ALGORITHM_T(name=algorithm_name))
        app.session.commit()

def find_or_create_parameter_id (app, algorithm_name,parameter_name):
    check_algorithm(app, algorithm_name)
    parameter=app.session.query(HFT_PARAMETERS_T).filter(and_(HFT_PARAMETERS_T.hft_algorithm_t_name==algorithm_name,
                                                       HFT_PARAMETERS_T.name==parameter_name)).first()
    if not parameter:
        parameter=HFT_PARAMETERS_T(hft_algorithm_t_name=algorithm_name,name=parameter_name)
        app.session.add(parameter)
        app.session.commit()
    return parameter.id

def convert_list(list):
    string_value='['
    first=True
    for value in list:
        if first:
            first=False
        else:
            string_value+=', '
        string_value+=str(value)
    string_value+=']'
    return string_value

def insert_hyperparameters_into_database(app,model_id,algorithm_name, treatment_id):
    for parameter_name in ALGORITHM_PARAMETERS[algorithm_name]:
        parameter_id=find_or_create_parameter_id(app,algorithm_name,parameter_name)
        value=getattr(models[algorithm_name].best_estimator_ , parameter_name)
        if type(value) is list:
            value = convert_list(value)
        app.session.add(HFT_MODEL_PARAMETERS_T(hft_model_id=model_id,treatment_id=treatment_id,
                                               hft_parameters_t_id=parameter_id,value=value))
    app.session.commit()

def insert_model_and_parameters_into_database(app,pickle_model,algorithm_name,treatment_id,dest,random_seed=10):
    model=HFT_MODEL_T(name=pickle_model,algorithm=algorithm_name,destination=dest,random_seed=random_seed)
    app.session.add(model)
    app.session.commit()
    model_id=model.id
    insert_hyperparameters_into_database(app,model_id, algorithm_name, treatment_id)    
    return model.id

def insert_model_metrics_into_database(app, pickle_model, algorithm_name, dest, random_seed=10):
    treatment_id = cluster_dfs[algorithm_name].iloc[0,1]
    model_id = insert_model_and_parameters_into_database(app,pickle_model,algorithm_name, treatment_id,dest)
    df_confusion_matrix=pd.DataFrame(confusion_matrix(y[algorithm_name], y_pred[algorithm_name]))
    THRESHOLD = int(threshold_dfs[algorithm_name].iloc[0,0])
    TN = int(df_confusion_matrix.iloc[0,0])
    FP = int(df_confusion_matrix.iloc[0,1])
    FN = int(df_confusion_matrix.iloc[1,0])
    TP = int(df_confusion_matrix.iloc[1,1])
    F1 = float(f1_score(y[algorithm_name], y_pred[algorithm_name], average='micro'))
    ACC = float(accuracy_score(y[algorithm_name], y_pred[algorithm_name]))
    OBS = int(len(cluster_dfs[algorithm_name].index))
    WEEKDAY = random_seed
    app.session.add(HFT_METRICS_T(f1_score = F1, true_negative = TN, true_positive = TP,
                                  false_negative = FN, false_positive = FP, accuracy = ACC,
                                  hft_model_id = model_id, hft_treatment_id = treatment_id, 
                                  threshold = THRESHOLD, number_of_observations = OBS, weekday = WEEKDAY))
    app.session.commit()

def save_pickle_and_metrics(app,folder):
    print("Step 7: Saving results")
    sub_folder='pkl_objects'
    dest = pickle_destination(folder,sub_folder)
    for cluster_code in cluster_dfs:
        pickle_model = cluster_code+'_'+'model.pkl'
        save_model_to_disk(dest,cluster_code)
        insert_model_metrics_into_database(app, pickle_model,cluster_code,dest)

def save_fitting_times(app,fitting_times_all_clusters, run_id):
    print(fitting_times_all_clusters)
    for cluster_code, fitting_time in fitting_times_all_clusters.items():
        treatment_id = cluster_dfs[cluster_code].iloc[0,1]
        app.session.add(HFT_FITTING_TIME_T(hft_treatment_id = str(treatment_id),
                                        algorithm = cluster_code, 
                                        fitting_time_sec = fitting_time,
                                        random_seed = 10,
                                        hft_run_id = run_id))
    app.session.commit()


def do_step_7(app):
    # In this step models are calculated per cluster
    # Each dataframe contains all the data of the participants in a cluster
    # The trainingset are the aggregation of the trainingdata used in individual calculation
    global cluster_dfs,threshold_dfs
    folder="results/clustering"
    start_logging(folder)
    run_id = get_run_id(app, 'Clustering on all characterisitics', 'test', 6, 'both')

    treatments_per_cluster = get_treatments_per_cluster(app)
    load_dataframe_and_thresholds(app, treatments_per_cluster)
    split_dataframes(treatments_per_cluster)
    fitting_times_all_clusters=fit_models()
    make_predictions()
    save_pickle_and_metrics(app, folder)
    save_fitting_times(app,fitting_times_all_clusters,run_id)
