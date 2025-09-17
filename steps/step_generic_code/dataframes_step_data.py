import pandas as pd
from sqlalchemy import and_, or_, not_
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from database.models.hft_tables import HFT_TREATMENT_T, HFT_DATA_T
from database.models.patient_data import THRESHOLDS

def add_columns_and_concat(results, steps):
    result = pd.DataFrame([row for row in steps],columns=['id','treatment_id','year','week','weekday','hour','sum_steps'] )
    if not result.empty:
        result['sum_steps_all'] = result.groupby(['year','week','weekday'])['sum_steps'].cumsum()
        result['daily_steps'] = result.groupby(['year','week','weekday'])['sum_steps_all'].transform('max')
        if results.empty:
            results = result
        else:
            results = pd.concat([results, result], ignore_index=True)
    return results

def get_dataframe(app):    
    print("Getting dataframe")
    results = pd.DataFrame()
    ids = app.session.query(HFT_TREATMENT_T.treatment_id).filter(HFT_TREATMENT_T.research_group.in_([1,2]))
    df_treatment_id = pd.DataFrame([row for row in ids], columns=['treatment_id'])
    for id in ids:
        steps = app.session.query(HFT_DATA_T.id, HFT_DATA_T.treatment_id, HFT_DATA_T.year, HFT_DATA_T.week, HFT_DATA_T.weekday,
                                    HFT_DATA_T.hour, HFT_DATA_T.steps)\
                            .join(HFT_TREATMENT_T, HFT_TREATMENT_T.treatment_id==HFT_DATA_T.treatment_id)\
                            .filter(and_(HFT_DATA_T.treatment_id==id[0],
                                         not_(HFT_DATA_T.weekday.in_([5,6])),
                                         HFT_DATA_T.year==2015,
                                         or_(and_(HFT_TREATMENT_T.research_group==2,HFT_DATA_T.week>15),
                                            and_(HFT_TREATMENT_T.research_group==1,HFT_DATA_T.week>4))))
        results = add_columns_and_concat(results, steps)
    results = results[(results['hour'].isin([7,8,9,10,11,12,13,14,15,16,17,18]))]
    results = results[~(results['daily_steps']==0)]
    results['sum_steps_hour'] = results.groupby(['treatment_id','year','week','weekday'])['sum_steps'].cumsum()
    return df_treatment_id, results

def get_dataframe_ns(app):
    print("Getting dataframe")
    results = pd.DataFrame()
    ids = app.session.query(HFT_TREATMENT_T.treatment_id).filter(HFT_TREATMENT_T.research_group.in_([3,4]))
    df_treatment_id = pd.DataFrame([row for row in ids], columns=['treatment_id'])
    for id in ids:
        steps = app.session.query(HFT_DATA_T.id, HFT_DATA_T.treatment_id, HFT_DATA_T.year, HFT_DATA_T.week, HFT_DATA_T.weekday,
                                  HFT_DATA_T.hour, HFT_DATA_T.steps).filter(HFT_DATA_T.treatment_id==id[0])
        results = add_columns_and_concat(results, steps)
    results = results[~(results['daily_steps']==0)]
    results['sum_steps_hour'] = results.groupby(['treatment_id','year','week','weekday'])['sum_steps'].cumsum()
    return df_treatment_id, results

def daily_steps_cat_f (steps_value,threshold):
    if (steps_value<threshold):
        return 0
    if (steps_value>=threshold):
        return 1
    
def get_cat(df_dataframe, df_threshold):
    x = df_dataframe['daily_steps']
    y = df_threshold['threshold']
    return np.vectorize(daily_steps_cat_f)(x, y)

def get_threshold(df_dataframe):
    print('Adding thresholds')
    df_steps = df_dataframe.groupby(['treatment_id','year','week','weekday'])['sum_steps_hour'].max()
    df_steps = df_steps.groupby(['treatment_id']).mean().reset_index(name='threshold')
    df_dataframe = pd.merge(df_dataframe, df_steps, left_on=['treatment_id'], right_on=['treatment_id'], how='outer')
    return df_dataframe

def get_threshold_ns(app, threshold_type, df_dataframe):
    print('Adding thresholds')
    threshold_result = app.session.query(THRESHOLDS.treatment_id, THRESHOLDS.year, THRESHOLDS.week, THRESHOLDS.weekday, THRESHOLDS.threshold)\
                                  .filter(and_(THRESHOLDS.threshold_type==threshold_type))
    threshold_df = pd.DataFrame([row for row in threshold_result], columns=['treatment_idth', 'yearth', 'weekth', 'weekdayth', 'threshold'])
    df_dataframe = df_dataframe.merge(threshold_df, left_on=['treatment_id', 'year', 'week', 'weekday'], 
                                     right_on=['treatment_idth', 'yearth', 'weekth', 'weekdayth'])
    df_dataframe = df_dataframe.drop(columns=['treatment_idth', 'yearth', 'weekth', 'weekdayth'])
    return df_dataframe

def get_full_sets(df_dataframe, variable_threshold=False, include_treatment=False):
    if variable_threshold:
        X = df_dataframe[['weekday','hour','sum_steps','sum_steps_hour','threshold']]
    else:
        # X = df_dataframe[['weekday','hour','sum_steps','sum_steps_hour']]
        X = df_dataframe[['hour','sum_steps','sum_steps_hour']]
    Y = pd.DataFrame()
    Y['dailysteps_cat'] = df_dataframe['dailysteps_cat']
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(np_scaled)
    if include_treatment:
        X['treatment_id']=df_dataframe['treatment_id']
        Y['treatment_id']=df_dataframe['treatment_id']
        X_scaled['treatment_id']=df_dataframe['treatment_id']
    return X, X_scaled, Y

def get_trainsets(df_dataframe, test_size, variable_threshold=False):
    X, X_scaled, Y = get_full_sets(df_dataframe, variable_threshold)
    X_train_s,X_test_s,Y_train_s,Y_test = train_test_split(X_scaled, Y, test_size=test_size, random_state=10)    
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=test_size, random_state=10)    
    return X_train_s,Y_train_s,X_train,Y_train

def get_train_and_testsets(df_dataframe, test_size, variable_threshold=False):
    if variable_threshold:
        X = df_dataframe[['weekday','hour','sum_steps','sum_steps_hour','threshold']]
    else:
        X = df_dataframe[['weekday','hour','sum_steps','sum_steps_hour']]
    Y = pd.DataFrame()
    Y['dailysteps_cat'] = df_dataframe['dailysteps_cat']
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(np_scaled)
    X_train_s,X_test_s,Y_train_s,Y_test = train_test_split(X_scaled, Y, test_size=test_size, random_state=10)    
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=test_size, random_state=10)    
    return X_train, Y_train, X_test, Y_test
