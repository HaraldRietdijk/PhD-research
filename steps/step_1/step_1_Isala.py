import glob, os   

import pandas as pd
import numpy as np

from database.models.hft_tables import HFT_TREATMENT_T
from database.models.isala_data import PATIENT_DERIVED
from database.models.measurement import IsalaID
from database.models.patient_data import ENUMLISTS, PATIENT_RESULTS, RESULT_MOMENTS, RESULT_TYPES
from steps.step_1.step_1_isala_conversion import CCI_WEIGHTS, CONVERSION_MATRIX, BOOLEAN_FIELDS, DATE_FIELDS, RESULTS_LIST
from steps.step_generic_code.enum_functions import get_enum_dict_values, get_enum_list_dicts

def add_derived_values(df):
    for col in CCI_WEIGHTS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({'ja':1,'nee':0}).fillna(0)

    df['CCI'] = sum(df[col]*w for col,w in CCI_WEIGHTS.items() if col in df.columns)
    df['LOS_THRESHOLD'] = df['Tumor_location'].apply(lambda x: 4 if x == 'Colon' else (5 if x == 'Rectum' else 7))
    df['los_prolonged'] = np.where(df['LOS'].fillna(0) > df['LOS_THRESHOLD'], 1, 0)
    df['Reoperation_bin'] = np.where(df['Reoperation'].isnull(),False,True)
    parts_d = [x for x in ['Readmissions_30days','los_prolonged'] if x in df.columns]
    parts_t = [x for x in ['Complication','ICU','Reoperation_bin','Readmissions_30days','los_prolonged'] if x in df.columns]
    df['non_textbook_outcome_d'] = (
        (df[parts_d].max(axis=1) == 1) | 
        (pd.to_numeric(df['Clavien_dindo'], errors='coerce') > 2)
    ).astype(int)
    df['non_textbook_outcome_t'] = (
        (df[parts_t].max(axis=1) == 1) | 
        (pd.to_numeric(df['Clavien_dindo'], errors='coerce') > 2)
    ).astype(int)
    return df

def get_dataframe(app, path):
    print("Reading files")
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    print("Creating dataframe")
    df = pd.concat(df_from_each_file, ignore_index=True)
    df.replace(np.nan, None, inplace=True)
    ENUM_LIST_DICTS = get_enum_list_dicts(app)
    replace_values = get_enum_dict_values(app, ENUM_LIST_DICTS['BOOLEAN'])
    for field in BOOLEAN_FIELDS:
        df[field].replace(replace_values, inplace=True)
    for field in DATE_FIELDS:
        df[field] =  pd.to_datetime(df[field], format='%m/%d/%Y')
        df.replace({pd.NaT: None}, inplace=True)
    df = add_derived_values(df)
    return df, df['pseudo_id'].unique()

def get_treatment_id(app, patient, patient_df):
        isala_id = app.session.query(IsalaID).filter(IsalaID.pseudo_id==patient).first()
        if isala_id:
            treatment_id = isala_id.treatment_id
        else:
            research_group = 7 if patient_df['Sort_prehabilitation'].iloc[0] in ['ecoach','eCoach'] else 6
            patient_df['gender'] = patient_df['gender'].map(lambda x: x.lower() if isinstance(x,str) else x)
            hft_treatment_id = HFT_TREATMENT_T(age = patient_df['Age'].iloc[0], 
                                               gender = patient_df['gender'].iloc[0],
                                               research_group = research_group)
            app.session.add(hft_treatment_id)
            app.session.commit()
            treatment_id = hft_treatment_id.treatment_id
            isala_id = IsalaID(pseudo_id = patient, treatment_id = treatment_id)
            app.session.add(isala_id)
            app.session.commit()
        return treatment_id

def get_enum_id(app, type, content):
    enum_row = app.session.query(ENUMLISTS).filter(ENUMLISTS.type==type,ENUMLISTS.content==content).first()
    if enum_row is None:
        print('adding: ',content)
        enum_row = ENUMLISTS(type=type, content=content, value_type=1)
        app.session.add(enum_row)
        app.session.commit()
    return enum_row.id

def store_table_fields(app, treatment_id, patient_df):
    print(treatment_id)
    for data_group, table_info in CONVERSION_MATRIX.items():
        print(data_group)
        model = table_info[0]
        model_row = app.session.query(model).filter(model.treatment_id==treatment_id).first()
        if model_row is None:
            model_row = model(treatment_id=treatment_id)
        for attributes in table_info[1]:
            if attributes[2]=='':
                setattr(model_row, attributes[1], patient_df[attributes[0]].iloc[0])
            else:
                if patient_df[attributes[0]].iloc[0] is None:
                    enum_id = None
                else:
                    enum_id = get_enum_id(app, attributes[2], patient_df[attributes[0]].iloc[0])
                setattr(model_row, attributes[1], enum_id)
        app.session.add(model_row)
    app.session.commit()

def get_result_id(app, result):
    result_type = app.session.query(RESULT_TYPES).filter(RESULT_TYPES.code==result).first()
    if result_type is None:
        result_type = RESULT_TYPES(code=result, description=result)
        app.session.add(result_type)
        app.session.commit()
    return result_type.id

def get_moment_id(app, moment):
    result_moment = app.session.query(RESULT_MOMENTS).filter(RESULT_MOMENTS.code==moment).first()
    if result_moment is None:
        result_moment = RESULT_TYPES(code=moment, description=moment)
        app.session.add(result_moment)
        app.session.commit()
    return result_moment.id

def store_result_fields(app, treatment_id, patient_df):
    for result in RESULTS_LIST:
        result_id = get_result_id(app, result[1])
        moment_id = get_moment_id(app, result[2])
        patient_result = app.session.query(PATIENT_RESULTS).filter(PATIENT_RESULTS.treatment_id==treatment_id,
                                                                   PATIENT_RESULTS.result_type==result_id, 
                                                                   PATIENT_RESULTS.result_moment==moment_id).first()
        if patient_result is None:
            patient_result = PATIENT_RESULTS(treatment_id=treatment_id, result_type=result_id, result_moment=moment_id)
        patient_result.float_value = patient_df[result[0]].iloc[0]
        app.session.add(patient_result)
        app.session.commit()

def store_derived_fields(app, treatment_id, patient_df):
    derived_fields =app.session.query(PATIENT_DERIVED).filter(PATIENT_DERIVED.treatment_id==treatment_id).first()
    if derived_fields is None:
        derived_fields = PATIENT_DERIVED(treatment_id=treatment_id)
    derived_fields.use_for_model = False if patient_df['Duration_surgery'].iloc[0] is None else True
    if derived_fields.use_for_model:
        derived_fields.cci = patient_df['CCI'].iloc[0]
        derived_fields.los_threshold = patient_df['LOS_THRESHOLD'].iloc[0]
        derived_fields.los_prolonged = patient_df['los_prolonged'].iloc[0]
        derived_fields.non_textbook_1 = patient_df['non_textbook_outcome_d'].iloc[0]
        derived_fields.non_textbook_2 = patient_df['non_textbook_outcome_t'].iloc[0]
    app.session.add(derived_fields)
    app.session.commit()


def store_data(app,df,patient_ids):
    for patient in patient_ids:
        patient_df = df[(df['pseudo_id']==patient)]
        treatment_id = get_treatment_id(app, patient, patient_df)
        store_table_fields(app, treatment_id, patient_df)
        store_result_fields(app, treatment_id, patient_df)
        store_derived_fields(app, treatment_id, patient_df)

def do_step_1_isala(app):
    path = r'.\data_origin_Isala'
    df, patient_ids = get_dataframe(app, path)
    store_data(app,df,patient_ids)