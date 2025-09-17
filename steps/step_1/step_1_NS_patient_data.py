import pandas as pd
import numpy as np
from sqlalchemy import and_

from database.models.hft_tables import HFT_TREATMENT_T
from database.models.measurement import AcceleroID
from database.models.patient_data import OPERATION_DATA, CLINICAL_PICTURE, PATIENT_INFO,\
                                         PATIENT_JOB, PATIENT_JOB_RECOVERY, PATIENT_PERSONAL, PATIENT_RESULTS
from steps.step_1.step_1_NS_meta_data import get_result_moments, get_result_types
from steps.step_generic_code.enum_functions import get_enum_list_dicts, get_enum_dict_values

def get_dataframe(app, file_name, hospital):
    def get_treatment_id(row):
        patient_id = hospital + str(row['nummer'])
        accellero_ID = app.session.query(AcceleroID.treatment_id).filter(AcceleroID.patient_id==patient_id).first()
        if accellero_ID is None:
            treatment_id = 0
            print('Patient not found:', patient_id)
        else:
            treatment_id = accellero_ID[0]
        return treatment_id

    print('Reading file ', file_name)
    df_from_patient_file = pd.read_csv(file_name)
    df_from_patient_file['treatment_id']=df_from_patient_file.apply(get_treatment_id, axis=1)
    df_from_patient_file.replace(np.nan, None, inplace=True)
    return df_from_patient_file

def update_treatment_data(app, df_from_patient_file, ENUM_LIST_DICTS):
    def update_record(row):
        treatment = app.session.query(HFT_TREATMENT_T).filter(HFT_TREATMENT_T.treatment_id==row['treatment_id']).first()
        treatment.age=row.iloc[2]
        treatment.gender=row.iloc[3]
        app.session.add(treatment)

    print("Updating treatment table")
    replace_values = get_enum_dict_values(app, ENUM_LIST_DICTS['GENDER'])
    df_from_patient_file.iloc[:,3].replace(replace_values, inplace=True)
    df_from_patient_file.loc[df_from_patient_file['treatment_id']>0].apply(update_record, axis=1)
    app.session.commit()

def update_clinical_picture(app, df_from_patient_file):
    def update_record(row):
        clinical_picture = app.session.query(CLINICAL_PICTURE).filter(CLINICAL_PICTURE.treatment_id==row['treatment_id']).first()
        if clinical_picture is None:
            clinical_picture = CLINICAL_PICTURE(treatment_id=row['treatment_id'])
        clinical_picture.work_capacity_preop = row.iloc[19]
        clinical_picture.number_of_days_of_work_preop = row.iloc[23]
        clinical_picture.number_of_problematic_days  =row.iloc[24]
        clinical_picture.amount_of_work_on_problematic_days = row.iloc[25]
        clinical_picture.problem_unpaid_workdays = row.iloc[26]
        clinical_picture.preop_work_capacity = row.iloc[33]
        app.session.add(clinical_picture)

    print("Updating clinical picture table")
    df_from_patient_file.loc[df_from_patient_file['treatment_id']>0].apply(update_record, axis=1)
    app.session.commit()

def update_operation_data(app, df_from_patient_file, ENUM_LIST_DICTS):
    def update_record(row):
        operation_data = app.session.query(OPERATION_DATA).filter(OPERATION_DATA.treatment_id==row['treatment_id']).first()
        if operation_data is None:
            operation_data = OPERATION_DATA(treatment_id=row['treatment_id'])
        operation_data.preop_screening_date=row.iloc[4]
        operation_data.operation_date=row.iloc[5]
        operation_data.operation_type=row.iloc[7]
        operation_data.operation_side=row.iloc[8]
        operation_data.los=row.iloc[9]
        app.session.add(operation_data)

    print("Updating operation data table")
    df_from_patient_file.iloc[:,7].replace(ENUM_LIST_DICTS['OPERTYPE'], inplace=True)
    replace_values = get_enum_dict_values(app, ENUM_LIST_DICTS['SIDE'])
    df_from_patient_file.iloc[:,8].replace(replace_values, inplace=True)
    df_from_patient_file.loc[df_from_patient_file['treatment_id']>0].apply(update_record, axis=1)
    app.session.commit()
            
def update_patient_info(app, df_from_patient_file, ENUM_LIST_DICTS):
    def update_record(row):
        patient_info = app.session.query(PATIENT_INFO).filter(PATIENT_INFO.treatment_id==row['treatment_id']).first()
        if patient_info is None:
            patient_info = PATIENT_INFO(treatment_id=row['treatment_id'])
        patient_info.kept_working=row.iloc[30]
        patient_info.additional_diseases=row.iloc[32]
        patient_info.working_capable_without_operation=row.iloc[34]
        patient_info.problems_caused_by_work=row.iloc[35]
        patient_info.remark=row.iloc[1]
        app.session.add(patient_info)

    print("Updating patient info table")
    replace_values = get_enum_dict_values(app, ENUM_LIST_DICTS['BOOLEAN'])
    df_from_patient_file.iloc[:,30].replace(replace_values, inplace=True)
    df_from_patient_file.iloc[:,34].replace(ENUM_LIST_DICTS['LIKELY'], inplace=True)
    df_from_patient_file.iloc[:,34].replace(np.nan, None, inplace=True)
    df_from_patient_file.iloc[:,35].replace(ENUM_LIST_DICTS['AGREE'], inplace=True)
    df_from_patient_file.iloc[:,35].replace(np.nan, None, inplace=True)
    df_from_patient_file.loc[df_from_patient_file['treatment_id']>0].apply(update_record, axis=1)
    app.session.commit()

def update_patient_job(app, df_from_patient_file, ENUM_LIST_DICTS):
    def update_record(row):
        patient_job = app.session.query(PATIENT_JOB).filter(PATIENT_JOB.treatment_id==row['treatment_id']).first()
        if patient_job is None:
            patient_job = PATIENT_JOB(treatment_id=row['treatment_id'])
        patient_job.type_of_work=row.iloc[18]
        patient_job.profession=row.iloc[20]
        patient_job.hours_per_week=row.iloc[21]
        patient_job.days_per_week=row.iloc[22]
        patient_job.breadwinner=row.iloc[29]
        patient_job.retirement_date=row.iloc[31]
        app.session.add(patient_job)

    print("Updating patient job table")
    replace_values = get_enum_dict_values(app, ENUM_LIST_DICTS['BOOLEAN'])
    df_from_patient_file.iloc[:,29].replace(replace_values, inplace=True)
    df_from_patient_file.iloc[:,18].replace(ENUM_LIST_DICTS['WORKTYPE'], inplace=True)
    df_from_patient_file.iloc[:,18].replace(np.nan, None, inplace=True)
    df_from_patient_file.loc[df_from_patient_file['treatment_id']>0].apply(update_record, axis=1)
    app.session.commit()

def update_patient_job_recovery(app, df_from_patient_file, ENUM_LIST_DICTS):
    def update_record(row):
        patient_job_recovery = app.session.query(PATIENT_JOB_RECOVERY).filter(PATIENT_JOB_RECOVERY.treatment_id==row['treatment_id']).first()
        if patient_job_recovery is None:
            patient_job_recovery = PATIENT_JOB_RECOVERY(treatment_id=row['treatment_id'])
        patient_job_recovery.return_to_work_date=row.iloc[10]
        patient_job_recovery.definitive_return_to_work_date=row.iloc[11]
        patient_job_recovery.expected_work_after_6_months=row.iloc[36]
        patient_job_recovery.working_after_6_months=row.iloc[37]
        patient_job_recovery.return_to_work_weeks=row.iloc[12]
        patient_job_recovery.definite_return_to_work_weeks=row.iloc[13]
        app.session.add(patient_job_recovery)

    print("Updating patient job recovery table")
    df_from_patient_file.iloc[:,37].replace(ENUM_LIST_DICTS['LIKELY'], inplace=True)
    df_from_patient_file.iloc[:,37].replace(np.nan, None, inplace=True)
    df_from_patient_file.loc[df_from_patient_file['treatment_id']>0].apply(update_record, axis=1)
    app.session.commit()

def update_patient_personal(app, df_from_patient_file, ENUM_LIST_DICTS):
    def update_record(row):
        patient_personal = app.session.query(PATIENT_PERSONAL).filter(PATIENT_PERSONAL.treatment_id==row['treatment_id']).first()
        if patient_personal is None:
            patient_personal = PATIENT_PERSONAL(treatment_id=row['treatment_id'])
        patient_personal.highest_education=row.iloc[17]
        patient_personal.height=row.iloc[27]
        patient_personal.weight=row.iloc[28]
        app.session.add(patient_personal)

    print("Updating patient personal table")
    df_from_patient_file.iloc[:,17].replace(ENUM_LIST_DICTS['HGO'], inplace=True)
    df_from_patient_file.iloc[:,17].replace(np.nan, None, inplace=True)
    df_from_patient_file.loc[df_from_patient_file['treatment_id']>0].apply(update_record, axis=1)
    app.session.commit()

def check_num_value(value, type):
    num_value = None
    if isinstance(value, int) or isinstance(value, float):
        num_value = value
    else:
        if not(value is None):
            try:
                if type=='int':
                    num_value = int(value)
                else:
                    num_value = float(value)
            except:
                print('Error on ' + value)
    return num_value

def update_patient_results(app, df_from_patient_file):
    def update_record(row):
        patient_result = app.session.query(PATIENT_RESULTS).filter(and_(PATIENT_RESULTS.treatment_id==row['treatment_id'],
                                                                        PATIENT_RESULTS.result_moment==result_moment,
                                                                        PATIENT_RESULTS.result_type==result_type))\
                                                           .first()
        if patient_result is None:
            patient_result = PATIENT_RESULTS(treatment_id=row['treatment_id'], result_type=result_type, result_moment=result_moment)
        if value_type=='int':
            patient_result.int_value = check_num_value(row.iloc[column_nr],'int')
        if value_type=='float':
            patient_result.float_value = check_num_value(row.iloc[column_nr],'float')
        else:
            patient_result.string_value = row.iloc[column_nr]
        app.session.add(patient_result)

    print("Updating results table")
    moments_dict = get_result_moments(app)
    types_dict = get_result_types(app)
    file_name = './data_origin_NS/source/Patient_results.csv'
    df_result_moments = pd.read_csv(file_name)
    rows = [tuple(x) for x in df_result_moments.values]
    cols = df_from_patient_file.shape[1]
    for row in rows:
        result_moment = moments_dict[row[4]]
        result_type = types_dict[row[3]]
        value_type = row[1]
        column_nr = row[2] 
        if (cols>column_nr): # ETZ has moments 4 and 5, NS does not
            df_from_patient_file.loc[df_from_patient_file['treatment_id']>0].apply(update_record, axis=1)
    app.session.commit()

def add_data_for_hospital(app, file_name, ENUM_LIST_DICTS, hospital):
    df_from_patient_file = get_dataframe(app, file_name, hospital)
    update_treatment_data(app, df_from_patient_file,ENUM_LIST_DICTS)
    update_clinical_picture(app, df_from_patient_file)
    update_operation_data(app, df_from_patient_file,ENUM_LIST_DICTS)
    update_patient_info(app, df_from_patient_file, ENUM_LIST_DICTS)
    update_patient_job(app, df_from_patient_file, ENUM_LIST_DICTS)
    update_patient_job_recovery(app, df_from_patient_file, ENUM_LIST_DICTS)
    update_patient_personal(app, df_from_patient_file, ENUM_LIST_DICTS)
    update_patient_results(app, df_from_patient_file)

def add_patient_data(app):
    ENUM_LIST_DICTS = get_enum_list_dicts(app)
    file_name = './data_origin_NS/source/patient_data_all.csv'
    add_data_for_hospital(app, file_name, ENUM_LIST_DICTS, 'NS')
    file_name = './data_origin_NS/source/patient_data_all_ETZ.csv'
    add_data_for_hospital(app, file_name, ENUM_LIST_DICTS, 'ETZ')