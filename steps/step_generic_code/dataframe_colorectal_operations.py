import pandas as pd
from sqlalchemy import and_
from database.models.hft_tables import HFT_TREATMENT_T
from database.models.isala_data import PATIENT_DERIVED
from database.models.patient_data import PATIENT_PERSONAL
from database.models.isala_data import PATIENT_INTAKE, PATIENT_CONDITION, PATIENT_HEALTH, OPERATION_DETAILS
from database.models.ns_views import PatientResults
from steps.step_1.step_1_isala_conversion import BOOLEAN_FIELDS
from steps.step_generic_code.dataframe_knee_operations import update_column
from steps.step_generic_code.enum_functions import get_enum_values_on_id
from sklearn.model_selection import train_test_split

SELECTED_FEATURES = {"Patient personal": (PATIENT_PERSONAL, [("BMI","bmi","")]),
                     "Patient intake" : (PATIENT_INTAKE,[("HADS","hads",""),("number_sessions_complete","number_sessions_complete",""),
                                                         ("PG_SGA_t0","pg_sga",""),("6MWT","six_mwt","6MWT"),("SRT_able","srt_able",""),
                                                         ("VSAQ","vsaq","")]),
                     "Patient condition" : (PATIENT_CONDITION,[("ASA","asa",""),("Surgery_asa","surgery_asa",""),
                                                               ("Anemia_treatment","anemia_treatment","")]),
                     "Patient health" : (PATIENT_HEALTH,[("Alcohol","alcohol","ALCOHOL"),("Corticosteroids","corticosteroids",""),
                                                         ("Drugs","drugs","DRUGS"),("Dyspnea_MRC","dyspnea_mrc",""),
                                                         ("Functioning","functioning","FUNCTION"), # removed ("KATZ","katz",""),
                                                         ("Personal_care","personal_care","PERSONCARE"),("Previous_cognitive","previous_cognitive",""),
                                                         ("Smoking","smoking","SMOKING"),("Two_stairs","two_stairs","")]),
                     "Operation details" : (OPERATION_DETAILS,[("Elective","elective","ELECTIVE"), # removed ("Surgeon","surgeon","SURGEON"),
                                                               ("Tumor_location","tumor_location","TUMORLOC"),
                                                               ("Type_anesthesia","type_anesthesia","ANESTH"),
                                                               ("Type_surgery","type_surgery","SURGERYTYP")])}

SELECTED_RESULTS = [("Hb_t0","Hb","T0",1), ("Albumin","Albumin","T0",1), ("Ferritin","Ferritin","T0",1), ("Transferrin","Transfer","T0",1),
                ("Trans_sat","Trans_sat","T0",1), ("CRP","CRP","T0",1), ("HbA1c","HbA1c","T0",1), ("Creatinine","Creatinine","T0",1),
                ("Hb_t1","Hb","T1",1), ("SRT_t0","SRT","T0",1), ("Handgrip_strength","Handgrip","T0",1), ("5TSTS","5XCRT","T0",1), 
                ("Second_rise","Secondrise","T0",1), ("6MWT_m","6MWT","T0",1), ("SRT_t1","SRT","T1",1), ("5TSTS_t1","5XCRT","T1",1),
                ("Secondrise_t1","Secondrise","T1",1), ("Handgrip_t1","Handgrip","T1",1)]


def add_table_features(app, selection_info, features_df, enum_values):
    model = selection_info[0]
    result = app.session.query(model).join(PATIENT_DERIVED,PATIENT_DERIVED.treatment_id==model.treatment_id)\
                        .filter(PATIENT_DERIVED.use_for_model==True).all()
    columns = ['treatment_id']
    columns.extend([field_info[1] for field_info in selection_info[1]])
    table_df = pd.DataFrame([[getattr(row, attr) for attr in columns]  for row in result], columns=columns)
    for field_info in selection_info[1]:
        if field_info[2]!="":
            table_df = update_column(table_df, field_info[1], 1, update_values=enum_values)
            table_df = update_column(table_df, field_info[1], 3)
        if field_info[0] in BOOLEAN_FIELDS:
            table_df = update_column(table_df, field_info[1], 3)
            table_df = update_column(table_df, field_info[1], 2)
        else:
            table_df = update_column(table_df, field_info[1], 3)
    return pd.merge(features_df, table_df, on='treatment_id', how='inner')

def add_result_features(app, selection_info, features_df):
    result = app.session.query(PatientResults).join(PATIENT_DERIVED,PATIENT_DERIVED.treatment_id==PatientResults.treatment_id)\
                        .filter(and_(PatientResults.result_type_code==selection_info[1],
                                     PatientResults.result_moment_code==selection_info[2],
                                     PATIENT_DERIVED.use_for_model==True)).all()
    value_columns = ['int_value','float_value','string_value']
    columns = ['treatment_id', value_columns[selection_info[3]]]
    column_name = selection_info[1]+'_'+selection_info[2]
    pd_colums = ['treatment_id', column_name]
    results_df = pd.DataFrame([[getattr(row, attr) for attr in columns]  for row in result], columns=pd_colums)
    results_df = update_column(results_df, column_name, 3)
    return pd.merge(features_df, results_df, on='treatment_id', how='inner')

def get_all_features(app):
    selected_patients = app.session.query(HFT_TREATMENT_T.treatment_id, PATIENT_DERIVED.non_textbook_1,PATIENT_DERIVED.non_textbook_2,
                                          HFT_TREATMENT_T.age,HFT_TREATMENT_T.gender,PATIENT_DERIVED.cci)\
                        .join(PATIENT_DERIVED,PATIENT_DERIVED.treatment_id==HFT_TREATMENT_T.treatment_id)\
                        .filter(PATIENT_DERIVED.use_for_model==True).all()
    features_df = pd.DataFrame([row for row in selected_patients], columns=['treatment_id', 'non_textbook_1',
                                                                             'non_textbook_2', 'age', 'gender', 'cci'])
    features_df = update_column(features_df, 'gender', 1, update_values={'m': 0, 'f': 1})
    enum_values = get_enum_values_on_id(app)
    for table, selection_info in SELECTED_FEATURES.items():
        features_df = add_table_features(app, selection_info, features_df, enum_values)
    for result_info in SELECTED_RESULTS:
        features_df = add_result_features(app, result_info, features_df)
    return features_df

def get_dataframe(app):
    print('Getting dataframe')
    features_df = get_all_features(app)
    dataframes = {}
    dataframes['Y'] = features_df['non_textbook_2']
    dataframes['Y_class'] = features_df['non_textbook_1']
    dataframes['X'] = features_df.iloc[:,3:]
    dataframes['X_train'], dataframes['X_test'], dataframes['Y_train'], dataframes['Y_test'] = \
            train_test_split(dataframes['X'],dataframes['Y'], test_size=0.4, random_state=42)
    _, _, dataframes['Y_class_train'], dataframes['Y_class_test'] = \
            train_test_split(dataframes['X'],dataframes['Y_class'], test_size=0.4, random_state=42)
    return dataframes
