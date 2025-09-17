import pandas as pd
import numpy as np
from sqlalchemy import and_

from database.models.patient_data import ENUMLISTS, RESULT_TYPES, RESULT_MOMENTS, RTW_CLASSES

def add_result_moments(app):
    file_name = './data_origin_NS/source/Result_moments.csv'
    df_result_moments = pd.read_csv(file_name)
    rows = [tuple(x) for x in df_result_moments.values]
    for row in rows:
        result_moment = app.session.query(RESULT_MOMENTS).filter(RESULT_MOMENTS.code==row[0]).first()
        if (result_moment is None):
            result_moment = RESULT_MOMENTS(code=row[0])
        result_moment.description = row[1]
        app.session.add(result_moment)
    app.session.commit()

def add_result_types(app):
    file_name = './data_origin_NS/source/Result_types.csv'
    df_result_types = pd.read_csv(file_name)
    rows = [tuple(x) for x in df_result_types.values]
    for row in rows:
        result_type = app.session.query(RESULT_TYPES).filter(RESULT_TYPES.code==row[0]).first()
        if (result_type is None):
            result_type = RESULT_TYPES(code=row[0])
        result_type.description = row[1]
        app.session.add(result_type)
    app.session.commit()

def get_result_moments(app):
    result_moments = app.session.query(RESULT_MOMENTS).all()
    moments_dict = {}
    for row in result_moments:
        moments_dict[row.code] = row.id
    return moments_dict

def get_result_types(app):
    result_types = app.session.query(RESULT_TYPES).all()
    types_dict = {}
    for row in result_types:
        types_dict[row.code] = row.id
    return types_dict

def add_enum_lists(app):
    file_name = './data_origin_NS/source/Enum_lists.csv'
    df_enum_lists = pd.read_csv(file_name)
    df_enum_lists.replace(np.nan, None, inplace=True)
    rows = [tuple(x) for x in df_enum_lists.values]
    for row in rows:
        enum_list = app.session.query(ENUMLISTS).filter(and_(ENUMLISTS.type==row[0], ENUMLISTS.content==row[1])).first()
        if (enum_list is None):
            enum_list = ENUMLISTS(type=row[0], content=row[1])
        enum_list.value_type = row[2]
        enum_list.int_value = row[3]
        enum_list.float_value = row[4]
        enum_list.bool_value = row[5]
        enum_list.string_value = row[6]
        app.session.add(enum_list)
    app.session.commit()

def add_rtw_classes(app):
    file_name = './data_origin_NS/source/rtw_classes.csv'
    df_rtw_classes_lists = pd.read_csv(file_name)
    rows = [tuple(x) for x in df_rtw_classes_lists.values]
    for row in rows:
        rtw_class = app.session.query(RTW_CLASSES).filter(and_(RTW_CLASSES.class_id==row[0],RTW_CLASSES.nr_of_classes==row[1])).first()
        if (rtw_class is None):
            rtw_class = RTW_CLASSES(class_id=row[0])
        rtw_class.nr_of_classes = row[1]
        rtw_class.from_week = row[2]
        rtw_class.to_week = row[3]
        app.session.add(rtw_class)
    app.session.commit()

def add_meta_data(app):
    add_result_moments(app)
    add_result_types(app)
    add_enum_lists(app)
    add_rtw_classes(app)
