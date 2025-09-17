from database.models.patient_data import ENUMLISTS

def get_enum_list_dicts(app):
    result_dicts = {}
    enum_lists = app.session.query(ENUMLISTS).all()
    for row in enum_lists:
        if not row.type in result_dicts:
            result_dicts[row.type]={}
        type_dict = result_dicts[row.type]
        type_dict[row.content]=row.id
    return result_dicts

def get_enum_dict_values(app, enum_dict):
    values_dict={}
    for key, value in enum_dict.items():
        record = app.session.query(ENUMLISTS).filter(ENUMLISTS.id==value).first()
        if record.value_type==1:
            values_dict[key]=record.int_value
        elif record.value_type==2:
            values_dict[key]=record.float_value
        elif record.value_type==3:
            values_dict[key]=record.bool_value
        else:
            values_dict[key]=record.string_value
    return values_dict

def get_enum_values_on_id(app):
    values_dict = {}
    enum_list_record = app.session.query(ENUMLISTS).all()
    for record in enum_list_record:
        if record.value_type==1:
            values_dict[record.id]=record.int_value
        elif record.value_type==2:
            values_dict[record.id]=record.float_value
        elif record.value_type==3:
            values_dict[record.id]=record.bool_value
        else:
            values_dict[record.id]=record.string_value
    return values_dict