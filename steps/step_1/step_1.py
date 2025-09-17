import glob, os   

import pandas as pd

from database.models.hft_tables import HFT_TREATMENT_T
from steps.step_1.step_1_generic import add_date_fields, remove_empty_steps, store_data_records, sum_per_hour
from steps.step_1.step_1_treatment_data import add_treatment_data

def get_dataframe_and_treatments(path):
    print("Reading files")
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    print("Creating dataframe")
    df = pd.concat(df_from_each_file, ignore_index=True)
    df['mdate'] = pd.to_datetime(df['date'])
    return df, df['treatment_id'].unique()

def add_treatment_id(treatment, treatment_df):
    treatment_df = treatment_df.reset_index()
    treatment_df['treatment_id']= treatment
    return treatment_df

def process_dataframe(app, df, treatment_ids):
    print("Processing dataframe")
    for treatment in treatment_ids:
        treatment_df = df[(df.treatment_id==treatment)].iloc[:,[6,8]]
        treatment_df = sum_per_hour(treatment_df)
        treatment_df = add_treatment_id(treatment, treatment_df)
        treatment_df = add_date_fields(treatment_df)
        treatment_df = remove_empty_steps(treatment_df)
        store_data_records(app, treatment_df)

def update_treatment_data(app, file_name):
    print("Updating treatment table")
    df_from_treatment_file = pd.read_csv(file_name)
    rows = [tuple(x) for x in df_from_treatment_file.values]
    for row in rows:
        treatment_id = row[3]
        treatment = app.session.query(HFT_TREATMENT_T).filter_by(treatment_id=treatment_id).first()
        if (treatment): 
            treatment.gender=row[2]
            treatment.research_group=row[1]
            app.session.add(treatment)
            app.session.commit()
        else:
            print('treatment_id not found:',treatment_id)

def do_step_1(app):
    path = r'.\data_origin'                # use your own path
    df, treatment_ids = get_dataframe_and_treatments(path)
    process_dataframe(app, df, treatment_ids)
    add_treatment_data(app)
