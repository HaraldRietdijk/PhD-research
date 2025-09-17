import glob, os   

import pandas as pd

from database.models.measurement import AcceleroID
from steps.step_1.step_1_generic import add_date_fields, remove_empty_steps, store_data_records, sum_per_hour
from steps.step_1.step_1_NS_meta_data import add_meta_data
from steps.step_1.step_1_NS_patient_data import add_patient_data
from steps.step_1.step_1_NS_thresholds import calculate_thresholds
from steps.step_1.step_1_NS_patient_activity import add_patient_activity_data

def get_dataframe_and_treatments(path):
    print("Reading files")
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    print("Creating dataframe")
    df = pd.concat(df_from_each_file, ignore_index=True)
    df = df.iloc[:,0:5]
    df['mdate'] = pd.to_datetime(df['timestamp'])
    return df, df['Set Nummer'].unique()

def add_treatment_id(app, patient, patient_df):
    treatment_id=app.session.query(AcceleroID.treatment_id).filter(AcceleroID.patient_id==patient).first()[0]
    patient_df = patient_df.reset_index()
    patient_df['treatment_id']= treatment_id
    return patient_df

def process_dataframe(app, df, patient_ids):
    print("Processing dataframe")
    for patient in patient_ids:
        patient_df = df[(df['Set Nummer']==patient)].iloc[:,[3,5]]
        patient_df = sum_per_hour(patient_df)
        patient_df = add_treatment_id(app, patient, patient_df)
        patient_df = add_date_fields(patient_df)
        patient_df = remove_empty_steps(patient_df)
        store_data_records(app, patient_df)
    return

def do_step_1_NS(app):
    path = r'.\data_origin_NS'
    df, patient_ids = get_dataframe_and_treatments(path)
    process_dataframe(app, df, patient_ids)
    add_meta_data(app)
    add_patient_data(app)
    add_patient_activity_data(app)
    calculate_thresholds(app)
