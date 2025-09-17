import glob, os   
import pandas as pd
import numpy as np  

from sqlalchemy import func
from database.models.measurement import AcceleroID, AcceleroData
from database.models.hft_tables import HFT_TREATMENT_T

def read_files(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
    print("Reading files")
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    print("Concatenating files")
    return pd.concat(df_from_each_file, ignore_index=True)

def store_ids(app, concatenated_df):
     for id in concatenated_df['Set Nummer'].unique():
         check_id = app.session.query(AcceleroID).filter(AcceleroID.patient_id==id).first()
         if check_id is None:
            if id.startswith("NS"):
                group=3
            else:
                group=4
            new_treatment = HFT_TREATMENT_T(research_group=group)
            app.session.add(new_treatment)
            app.session.commit()
            new_id = AcceleroID(treatment_id=new_treatment.treatment_id, patient_id=id)
            app.session.add(new_id)
            app.session.commit()

def add_rows(app, concatenated_df):
    print("Adding rows")
    rows = [tuple(x) for x in concatenated_df.values]
    counter = 0
    for row in rows:
        counter+=1
        app.session.add(AcceleroData(patient_id = row[4],
                                     datetime = row[0],
                                     score = row[1],
                                     mets = row[2],
                                     steps=row[3]))
        if counter%50000==0:
            print("Added: ",counter)
            app.session.commit()            
    app.session.commit()
    return counter
          
def do_read_raw_data_NS(app):
    path = r'.\data_origin_NS'                # use your own path
    concatenated_df = read_files(path)
    store_ids(app, concatenated_df)
    counter = add_rows(app, concatenated_df)
    print("Processed", counter)
    print("Added :", app.session.query(func.count(AcceleroData.id))[0][0])