import glob, os   
import pandas as pd
import numpy as np  

from sqlalchemy import func
from database.models.measurement import FitbitData
from database.models.hft_tables import HFT_TREATMENT_T

def read_files(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
    print("Reading files")
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    print("Concatenating files")
    return pd.concat(df_from_each_file, ignore_index=True)

def replace_space(concatenated_df):
    print("Replacing spaces")
    concatenated_df['steps'].replace(['',np.nan],0, inplace=True)
    concatenated_df['mets'].replace(['',np.nan],0, inplace=True)
    concatenated_df['calories'].replace(['',np.nan],0, inplace=True)
    concatenated_df['level'].replace(['',np.nan],0, inplace=True)
    concatenated_df['distance'].replace(['',np.nan],0, inplace=True)
    return concatenated_df

def store_ids(app, concatenated_df):
    for treatment_id in concatenated_df['treatment_id'].unique():
        treatment = HFT_TREATMENT_T(treatment_id=treatment_id)
        app.session.add(treatment)
    app.session.commit()

def add_rows(app, concatenated_df):
    print("Adding rows")
    rows = [tuple(x) for x in concatenated_df.values]
    counter = 0
    for row in rows:
        counter+=1
        app.session.add(FitbitData(treatment_id = row[0],
                                   fitbit_id = row[1],
                                   datetime = row[2],
                                   calories = row[3],
                                   mets = row[4],
                                   level = row[5],
                                   steps = row[6],
                                   distance = row[7]))
        if counter%50000==0:
                print("Added: ",counter)
                app.session.commit()            
    app.session.commit()
    return counter
          
def do_read_raw_data(app):
    path = r'.\data_origin'                # use your own path
    concatenated_df = read_files(path)
    concatenated_df = replace_space(concatenated_df)
    store_ids(app, concatenated_df)
    counter = add_rows(app, concatenated_df)
    print("Processed", counter)
    print("Added :", app.session.query(func.count(FitbitData.id))[0][0])