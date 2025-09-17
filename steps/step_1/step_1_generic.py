import numpy as np

from database.models.hft_tables import HFT_DATA_T

def sum_per_hour(patient_df):
    patient_df = patient_df.set_index('mdate')
    return patient_df.resample('H').sum()

def add_date_fields(patient_df):
    patient_df['year'] = patient_df['mdate'].dt.year
    patient_df['week'] = patient_df['mdate'].dt.isocalendar().week
    patient_df['weekday'] = patient_df['mdate'].dt.dayofweek
    patient_df['hour'] = patient_df['mdate'].dt.hour
    patient_df['date'] = patient_df['mdate'].dt.date
    return patient_df

def remove_empty_steps(patient_df):
    patient_df['steps'].replace('',np.nan, inplace=True)
    patient_df.dropna(subset=['steps'], inplace=True)
    return patient_df

def store_data_records(app, patient_df):
    rows = [tuple(x) for x in patient_df.values]
    for row in rows:
        if (row[7].month==1 and row[4]>51):
            year = row[3]-1
        else:
            year = row[3]    
        data_rec = HFT_DATA_T(treatment_id=row[2], mdate=row[7], year=year, week=row[4],
                        weekday=row[5], hour=row[6], steps=row[1])
        app.session.add(data_rec)
    app.session.commit()
