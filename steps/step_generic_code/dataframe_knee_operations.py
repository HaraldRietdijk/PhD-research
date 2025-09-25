import random
import pandas as pd
import numpy as np
from sqlalchemy import and_
from database.models.hft_tables import HFT_TREATMENT_T
from database.models.patient_data import CLINICAL_PICTURE, PATIENT_INFO, PATIENT_JOB_RECOVERY, PATIENT_PERSONAL,\
                                         OPERATION_DATA, PATIENT_JOB, PATIENT_ACTIVITY
from database.models.ns_views import PatientResults, PatientRTWClass
from steps.step_generic_code.enum_functions import get_enum_values_on_id
from sklearn.model_selection import train_test_split

def update_column(features_df, column_name, update_type, update_values=0):
    if update_type==0: # Impute, replace None with constant
        features_df[column_name].iloc[:].replace(np.nan, update_values, inplace=True)
    elif update_type==1: # Replace string content with values list
        features_df[column_name].iloc[:].replace(update_values, inplace=True)
    elif update_type==2: # replace boolean
        features_df[column_name]= features_df[column_name].astype(int)
    return features_df

def add_result_column(app, result, features_df, result_type, moment, value_type, update_type, update_values):
    column_query = app.session.query(HFT_TREATMENT_T.treatment_id, PatientResults.int_value
                                       , PatientResults.float_value, PatientResults.string_value)\
                        .filter(and_(HFT_TREATMENT_T.treatment_id==PatientResults.treatment_id, 
                                     PatientResults.result_type_code==result_type, PatientResults.result_moment_code==moment))\
                        .subquery()
    results_sq = result.subquery()
    if value_type=='int':
        join_results = app.session.query(results_sq, column_query.c.int_value).join(column_query, column_query.c.treatment_id==results_sq.c.treatment_id)
    elif value_type=='float':
        join_results = app.session.query(results_sq, column_query.c.float_value).join(column_query, column_query.c.treatment_id==results_sq.c.treatment_id)
    elif value_type=='string':
        join_results = app.session.query(results_sq, column_query.c.string_value).join(column_query, column_query.c.treatment_id==results_sq.c.treatment_id)
    column_name = result_type + '_' + moment 
    features_df[column_name] = [row[-1] for row in join_results]
    features_df = update_column(features_df, column_name, update_type, update_values)
    return features_df

def get_all_features(app, nr_of_classes):
    print('Step 10: Getting dataframes for ' + str(nr_of_classes) + ' classes')
    result = app.session.query(CLINICAL_PICTURE.treatment_id, PATIENT_JOB_RECOVERY.definite_return_to_work_weeks,
                               PatientRTWClass.class_id, OPERATION_DATA.operation_type, OPERATION_DATA.los,
                               PATIENT_JOB.hours_per_week, PATIENT_JOB.type_of_work, PATIENT_JOB.breadwinner,
                               CLINICAL_PICTURE.number_of_problematic_days, CLINICAL_PICTURE.amount_of_work_on_problematic_days,
                               CLINICAL_PICTURE.problem_unpaid_workdays, CLINICAL_PICTURE.preop_work_capacity, 
                               CLINICAL_PICTURE.work_capacity_preop, CLINICAL_PICTURE.number_of_days_of_work_preop, 
                               HFT_TREATMENT_T.age, PATIENT_PERSONAL.height, PATIENT_PERSONAL.weight, 
                               PATIENT_INFO.working_capable_without_operation, PATIENT_INFO.problems_caused_by_work,
                               HFT_TREATMENT_T.gender, HFT_TREATMENT_T.research_group-3,
                               PATIENT_ACTIVITY.average_inactivity_pre, PATIENT_ACTIVITY.average_inactivity_post,
                               PATIENT_ACTIVITY.average_steps_pre, PATIENT_ACTIVITY.average_steps_post)\
                                .filter(and_(CLINICAL_PICTURE.treatment_id == PATIENT_JOB_RECOVERY.treatment_id,
                                             CLINICAL_PICTURE.treatment_id == OPERATION_DATA.treatment_id,
                                             CLINICAL_PICTURE.treatment_id == PatientRTWClass.treatment_id,
                                             PatientRTWClass.nr_of_classes == nr_of_classes,
                                             CLINICAL_PICTURE.treatment_id == HFT_TREATMENT_T.treatment_id,
                                             CLINICAL_PICTURE.treatment_id == PATIENT_PERSONAL.treatment_id,
                                             CLINICAL_PICTURE.treatment_id == PATIENT_INFO.treatment_id,
                                             CLINICAL_PICTURE.treatment_id == PATIENT_JOB.treatment_id,
                                             CLINICAL_PICTURE.treatment_id == PATIENT_ACTIVITY.treatment_id,
                                             PATIENT_JOB_RECOVERY.definite_return_to_work_weeks != None,
                                             PATIENT_PERSONAL.height != None,
                                             PATIENT_JOB_RECOVERY.definite_return_to_work_weeks<60
                                                 ))
    features_df = pd.DataFrame([row for row in result],
                          columns=['treatment_id', 'def_ret_work_weeks', 'class_id', 'operation_type', 'days_in_hospital',
                                   'hours_per_week', 'type_of_work', 'breadwinner', 'nr_prob_days', 
                                   'am_work_prob_days', 'prob_unpaid_days', 'preop_capa', 'Work_capacity_preop', 
                                   'nr_of_days_off_work', 'age', 'height', 'weight', 'work_capable_wo_oper', 
                                   'problems_caused_work', 'gender', 'hospital', 
                                   'average_inactivity_pre', 'average_inactivity_post', 
                                   'average_steps_pre', 'average_steps_post'])

    enum_values = get_enum_values_on_id(app)
    features_df = update_column(features_df, 'operation_type', 1, update_values=enum_values)
    features_df = update_column(features_df, 'operation_type', 0, update_values=0)
    features_df = update_column(features_df, 'days_in_hospital', 0, update_values=2)
    features_df = update_column(features_df, 'hours_per_week', 0, update_values=40)
    features_df = update_column(features_df, 'type_of_work', 1, update_values=enum_values)
    features_df = update_column(features_df, 'type_of_work', 0, update_values=1)
    features_df = update_column(features_df, 'breadwinner', 0, update_values=1)
    features_df = update_column(features_df, 'breadwinner', 2)
    features_df = update_column(features_df, 'breadwinner', 0, update_values=1)
    features_df = update_column(features_df, 'nr_prob_days', 0, update_values=0)
    features_df = update_column(features_df, 'am_work_prob_days', 0, update_values=100)
    features_df = update_column(features_df, 'prob_unpaid_days', 0, update_values=0)    
    features_df = update_column(features_df, 'preop_capa', 0, update_values=10)    
    features_df = update_column(features_df, 'Work_capacity_preop', 0, update_values=100)    
    features_df = update_column(features_df, 'nr_of_days_off_work', 0, update_values=0)    
    features_df = update_column(features_df, 'work_capable_wo_oper', 1, update_values=enum_values)
    features_df = update_column(features_df, 'problems_caused_work', 1, update_values=enum_values)
    features_df = update_column(features_df, 'gender', 1, update_values={'m': 0, 'f': 1})

    features_df = add_result_column(app, result, features_df,  '30CRT', 'T0', 'int', 0, 0)
    features_df = add_result_column(app, result, features_df, '5XCRT', 'T0', 'float', 0, 0)
    features_df = add_result_column(app, result, features_df, '6MWT', 'T0', 'int', 0, 0)
    features_df = add_result_column(app, result, features_df, 'LFTT', 'T0', 'float', 0, 0)
    features_df = add_result_column(app, result, features_df, 'CD10', 'T0', 'int', 0, 9)
    features_df = add_result_column(app, result, features_df, 'CSI', 'T0', 'int', 0, 24)
    features_df = add_result_column(app, result, features_df, 'WORQ', 'T0', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_A', 'T0', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_P', 'T0', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_Q', 'T0', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_S', 'T0', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_SP', 'T0', 'float', 0, 50)

    features_df = add_result_column(app, result, features_df, '30CRT', 'T1', 'int', 0, 0)
    features_df = add_result_column(app, result, features_df, '5XCRT', 'T1', 'float', 0, 0)
    features_df = add_result_column(app, result, features_df, '6MWT', 'T1', 'int', 0, 0)
    features_df = add_result_column(app, result, features_df, 'CSI', 'T1', 'int', 0, 24)
    features_df = add_result_column(app, result, features_df, 'WORQ', 'T1', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_A', 'T1', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_P', 'T1', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_Q', 'T1', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_S', 'T1', 'float', 0, 50)
    features_df = add_result_column(app, result, features_df, 'KOOS_SP', 'T1', 'float', 0, 50)
    return features_df

def get_dataframe(app, nr_of_classes):
    features_df = get_all_features(app, nr_of_classes)
    dataframes = {}
    dataframes['Y'] = features_df.iloc[:,1]
    dataframes['Y_class'] = features_df.iloc[:,2]
    dataframes['X'] = features_df.iloc[:,3:]
    dataframes['X_train'], dataframes['X_test'], dataframes['Y_train'], dataframes['Y_test'] = \
            train_test_split(dataframes['X'],dataframes['Y'], test_size=0.3, random_state=42)
    _, _, dataframes['Y_class_train'], dataframes['Y_class_test'] = \
            train_test_split(dataframes['X'],dataframes['Y_class'], test_size=0.3, random_state=42)
    return dataframes
