from sqlalchemy import and_, func
import pandas as pd
import datetime

from database.models.patient_data import CLINICAL_PICTURE, PATIENT_JOB_RECOVERY, PATIENT_PERSONAL, OPERATION_DATA, PATIENT_ACTIVITY
from database.models.measurement import AcceleroData, AcceleroID

def get_accelerator_ids(app):
    results = app.session.query(CLINICAL_PICTURE.treatment_id, AcceleroID.patient_id)\
                 .join(PATIENT_JOB_RECOVERY, PATIENT_JOB_RECOVERY.treatment_id==CLINICAL_PICTURE.treatment_id)\
                 .join(PATIENT_PERSONAL, PATIENT_PERSONAL.treatment_id==CLINICAL_PICTURE.treatment_id)\
                 .join(AcceleroID, AcceleroID.treatment_id==CLINICAL_PICTURE.treatment_id)\
                 .filter(and_(PATIENT_JOB_RECOVERY.definite_return_to_work_weeks != None,
                              PATIENT_PERSONAL.height != None,
                              PATIENT_JOB_RECOVERY.definite_return_to_work_weeks<60))\
                 .all()
    accellero_ids = [(row[0],row[1]) for row in results]
    return accellero_ids

def add_date_fields(accelero_data):
    accelero_data['year'] = accelero_data['datetime'].dt.year.astype(int)
    accelero_data['week'] = accelero_data['datetime'].dt.isocalendar().week.astype(int)
    accelero_data['weekday'] = accelero_data['datetime'].dt.day_of_week.astype(int)
    accelero_data['hour'] = accelero_data['datetime'].dt.hour.astype(int)
    accelero_data['minute'] = accelero_data['datetime'].dt.minute.astype(int)
    accelero_data.iloc[:,7].replace({15 : 1, 30 : 2, 45: 3}, inplace=True)
    return accelero_data

def drop_early_and_late_hours(accelero_data):
    accelero_data.drop(accelero_data[accelero_data.hour < 7].index, inplace=True)
    accelero_data.drop(accelero_data[accelero_data.hour >=22].index, inplace=True)
    return accelero_data

def get_accellero_data_pre_operation_for_patient(app, patient_id):
    results_pre = app.session.query(AcceleroData.datetime, AcceleroData.mets, AcceleroData.steps)\
                         .join(AcceleroID, AcceleroID.patient_id == AcceleroData.patient_id)\
                         .join(OPERATION_DATA, AcceleroID.treatment_id == OPERATION_DATA.treatment_id)\
                         .filter(and_(AcceleroData.patient_id==patient_id, AcceleroData.datetime<OPERATION_DATA.operation_date))\
                         .all()
    if (len(results_pre)==0):
        accelero_data = pd.DataFrame()
    else:
        accelero_data_pre = pd.DataFrame([row for row in results_pre], columns=['datetime', 'mets', 'steps'])
        accelero_data = add_date_fields(accelero_data_pre)
        accelero_data = drop_early_and_late_hours(accelero_data)
    return accelero_data

def get_accellero_data_post_operation_for_patient(app, patient_id):
    results_post = app.session.query(OPERATION_DATA.operation_date, AcceleroData.datetime, AcceleroData.mets, AcceleroData.steps)\
                         .join(AcceleroID, AcceleroID.patient_id == AcceleroData.patient_id)\
                         .join(OPERATION_DATA, AcceleroID.treatment_id == OPERATION_DATA.treatment_id)\
                         .filter(and_(AcceleroData.patient_id==patient_id, 
                                      AcceleroData.datetime>OPERATION_DATA.operation_date))\
                         .all()
    if (len(results_post)==0):
        accelero_data = pd.DataFrame()
    else:
        accelero_data_post = pd.DataFrame([row for row in results_post], columns=['operation_date', 'datetime', 'mets', 'steps'])
        accelero_data_post.drop(accelero_data_post[accelero_data_post.datetime < (accelero_data_post.operation_date + datetime.timedelta(days=43))].index, inplace=True)
        accelero_data = accelero_data_post.iloc[:,1:]
        accelero_data = add_date_fields(accelero_data)
        accelero_data = drop_early_and_late_hours(accelero_data)
    return accelero_data

def set_last_values(row):
    last_values = {}
    last_values['last_day'] = row.weekday
    last_values['last_week'] = row.week
    last_values['last_hour'] = row.hour
    last_values['last_minute'] = row.minute
    return last_values

def check_morning_activity(row):
    inactivity_count = 0
    if row.hour > 9:
        inactivity_count = 1
    return inactivity_count

def check_evening_activity(row):
    inactivity_count = 0
    if row.hour < 20:
        inactivity_count = 1
    return inactivity_count

def get_inactivity_count_list(accelero_data):
    first = True
    inactivity_counts_per_day = []
    inactivity_count = 0
    for index, row in accelero_data.iterrows():
        if first:
            last_values = set_last_values(row)
            first = False
            inactivity_count = check_morning_activity(row)
        else:
            if (row.weekday != last_values['last_day']) or (row.week != last_values['last_week']):
                inactivity_count += check_evening_activity(row)
                inactivity_counts_per_day.append(inactivity_count)
                inactivity_count = check_morning_activity(row)
            else:
                last_movement = last_values['last_hour']*4 + last_values['last_minute']
                next_movement = row.hour*4 + row.minute
                if next_movement-last_movement>2:
                    inactivity_count += (next_movement-last_movement-1)//2
            last_values = set_last_values(row)
    inactivity_count += check_evening_activity(row)
    inactivity_counts_per_day.append(inactivity_count)
    return inactivity_counts_per_day

def get_inactivity_count(accelero_data):
    if accelero_data.empty:
        inactivity_counts = [9999]
    else:
        inactivity_counts = get_inactivity_count_list(accelero_data)
    return sum(inactivity_counts)/len(inactivity_counts)

def store_patient_inactivity(app, treatment_id, averages):
    patient_activity = app.session.query(PATIENT_ACTIVITY).filter(PATIENT_ACTIVITY.treatment_id==treatment_id).first()
    if not patient_activity:
        patient_activity = PATIENT_ACTIVITY(treatment_id=treatment_id)
    patient_activity.average_inactivity_pre = averages['average_inactivity_pre']
    patient_activity.average_inactivity_post = averages['average_inactivity_post']
    patient_activity.average_steps_pre = averages['average_steps_pre']
    patient_activity.average_steps_post = averages['average_steps_post']
    app.session.add(patient_activity)
    app.session.commit()

def impute_average_inactivity_pre(app):
    missing_averages = app.session.query(PATIENT_ACTIVITY)\
                          .filter(and_(PATIENT_ACTIVITY.average_inactivity_pre==9999, 
                                       PATIENT_ACTIVITY.average_inactivity_post<9999))\
                          .all()
    for missing_average in missing_averages:
        average_post = missing_average.average_inactivity_post
        neighbours = app.session.query(PATIENT_ACTIVITY.average_inactivity_pre, func.abs(PATIENT_ACTIVITY.average_inactivity_post-average_post))\
                                .filter(PATIENT_ACTIVITY.average_inactivity_pre<9999)\
                                .order_by(func.abs(PATIENT_ACTIVITY.average_inactivity_post-average_post)).limit(7)
        neighbour_list = [row[0] for row in neighbours]
        impute_average = sum(neighbour_list)/len(neighbour_list)
        missing_average.average_inactivity_pre = impute_average
        app.session.add(missing_average)
        app.session.commit()

def impute_average_inactivity_post(app):
    missing_averages = app.session.query(PATIENT_ACTIVITY)\
                          .filter(and_(PATIENT_ACTIVITY.average_inactivity_post==9999, 
                                       PATIENT_ACTIVITY.average_inactivity_pre<9999))\
                          .all()
    for missing_average in missing_averages:
        average_pre = missing_average.average_inactivity_pre
        neighbours = app.session.query(PATIENT_ACTIVITY.average_inactivity_post, func.abs(PATIENT_ACTIVITY.average_inactivity_pre-average_pre))\
                                .filter(PATIENT_ACTIVITY.average_inactivity_post<9999)\
                                .order_by(func.abs(PATIENT_ACTIVITY.average_inactivity_pre-average_pre)).limit(7)
        neighbour_list = [row[0] for row in neighbours]
        impute_average = sum(neighbour_list)/len(neighbour_list)
        missing_average.average_inactivity_post = impute_average
        app.session.add(missing_average)
        app.session.commit()

def impute_average_inactivity_both(app):
    averages = app.session.query(func.avg(PATIENT_ACTIVITY.average_inactivity_pre), 
                                 func.avg(PATIENT_ACTIVITY.average_inactivity_post))\
                          .filter(PATIENT_ACTIVITY.average_inactivity_post<9999).all()[0]
    missing_averages = app.session.query(PATIENT_ACTIVITY)\
                          .filter(PATIENT_ACTIVITY.average_inactivity_post==9999)\
                          .all()
    for missing_average in missing_averages:
        missing_average.average_inactivity_pre = averages[0]
        missing_average.average_inactivity_post = averages[1]
        app.session.add(missing_average)
        app.session.commit()

def impute_average_steps_pre(app):
    missing_averages = app.session.query(PATIENT_ACTIVITY)\
                          .filter(and_(PATIENT_ACTIVITY.average_steps_pre==99999, 
                                       PATIENT_ACTIVITY.average_steps_post<99999))\
                          .all()
    for missing_average in missing_averages:
        average_post = missing_average.average_inactivity_post
        neighbours = app.session.query(PATIENT_ACTIVITY.average_steps_pre, func.abs(PATIENT_ACTIVITY.average_steps_post-average_post))\
                                .filter(PATIENT_ACTIVITY.average_steps_pre<99999)\
                                .order_by(func.abs(PATIENT_ACTIVITY.average_steps_post-average_post)).limit(7)
        neighbour_list = [row[0] for row in neighbours]
        impute_average = sum(neighbour_list)/len(neighbour_list)
        missing_average.average_steps_pre = impute_average
        app.session.add(missing_average)
        app.session.commit()

def impute_average_steps_post(app):
    missing_averages = app.session.query(PATIENT_ACTIVITY)\
                          .filter(and_(PATIENT_ACTIVITY.average_steps_post==99999, 
                                       PATIENT_ACTIVITY.average_steps_pre<99999))\
                          .all()
    for missing_average in missing_averages:
        average_pre = missing_average.average_steps_pre
        neighbours = app.session.query(PATIENT_ACTIVITY.average_steps_post, func.abs(PATIENT_ACTIVITY.average_steps_pre-average_pre))\
                                .filter(PATIENT_ACTIVITY.average_steps_post<99999)\
                                .order_by(func.abs(PATIENT_ACTIVITY.average_steps_pre-average_pre)).limit(7)
        neighbour_list = [row[0] for row in neighbours]
        impute_average = sum(neighbour_list)/len(neighbour_list)
        missing_average.average_steps_post = impute_average
        app.session.add(missing_average)
        app.session.commit()

def impute_average_steps_both(app):
    averages = app.session.query(func.avg(PATIENT_ACTIVITY.average_steps_pre), 
                                 func.avg(PATIENT_ACTIVITY.average_steps_post))\
                          .filter(PATIENT_ACTIVITY.average_steps_post<99999).all()[0]
    missing_averages = app.session.query(PATIENT_ACTIVITY)\
                          .filter(PATIENT_ACTIVITY.average_steps_post==99999)\
                          .all()
    for missing_average in missing_averages:
        missing_average.average_steps_pre = averages[0]
        missing_average.average_steps_post = averages[1]
        app.session.add(missing_average)
        app.session.commit()

def get_steps_average(accelero_data):
    if accelero_data.empty:
        average_steps = [99999]
    else:
        average_steps = []
        df_per_day=accelero_data.groupby(['year','week', 'weekday'])
        for name, groups in df_per_day:
            average_steps.append(groups["steps"].sum())
    return sum(average_steps)/len(average_steps)

def add_patient_activity_data(app):
    accelerator_ids = get_accelerator_ids(app)
    for treatment_id, patient_id in accelerator_ids:
        averages={}
        accelero_data = get_accellero_data_pre_operation_for_patient(app, patient_id)
        averages['average_inactivity_pre'] = get_inactivity_count(accelero_data)
        averages['average_steps_pre'] = get_steps_average(accelero_data)
        accelero_data = get_accellero_data_post_operation_for_patient(app, patient_id)
        averages['average_inactivity_post'] = get_inactivity_count(accelero_data)
        averages['average_steps_post'] = get_steps_average(accelero_data)
        store_patient_inactivity(app, treatment_id, averages)
    impute_average_inactivity_pre(app)
    impute_average_steps_pre(app)
    impute_average_inactivity_post(app)
    impute_average_steps_post(app)
    impute_average_inactivity_both(app)
    impute_average_steps_both(app)
