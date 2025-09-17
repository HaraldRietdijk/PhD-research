from sqlalchemy import Column, ForeignKey, VARCHAR, Integer, DateTime, Float, Boolean
from database.models.model import DBActivityModel
from database.models.hft_tables import HFT_TREATMENT_T
from datetime import datetime

class ENUMLISTS(DBActivityModel):

    __tablename__ = "enum_lists"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    type = Column(VARCHAR(10), nullable=False)
    content = Column(VARCHAR(256, collation='utf8_bin'), nullable=False)
    value_type = Column(Integer, nullable=False)
    int_value = Column(Integer)
    float_value = Column(Float)
    bool_value = Column(Boolean)
    string_value = Column(VARCHAR(10))    

class OPERATION_DATA(DBActivityModel):

    __tablename__ = "operation_data"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    preop_screening_date = Column(DateTime)
    operation_date = Column(DateTime)
    los = Column(Integer)
    operation_type = Column(Integer, ForeignKey(ENUMLISTS.id))
    operation_side = Column(VARCHAR(1))

class PATIENT_JOB_RECOVERY(DBActivityModel):

    __tablename__ = "patient_job_recovery"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    return_to_work_date = Column(DateTime)
    definitive_return_to_work_date = Column(DateTime)
    expected_work_after_6_months = Column(Float)
    working_after_6_months = Column(Integer, ForeignKey(ENUMLISTS.id))
    return_to_work_weeks = Column(Float)
    definite_return_to_work_weeks = Column(Float)

class PATIENT_INFO(DBActivityModel):

    __tablename__ = "patient_info"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    kept_working = Column(Boolean)
    additional_diseases = Column(VARCHAR(256))   
    working_capable_without_operation = Column(Integer, ForeignKey(ENUMLISTS.id))
    problems_caused_by_work = Column(Integer, ForeignKey(ENUMLISTS.id))
    remark = Column(VARCHAR(1024))

class PATIENT_JOB(DBActivityModel):

    __tablename__ = "patient_job"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    type_of_work= Column(Integer, ForeignKey(ENUMLISTS.id))
    profession = Column(VARCHAR(100))
    hours_per_week = Column(Integer)
    days_per_week = Column(Float)
    breadwinner = Column(Boolean)
    retirement_date = Column(DateTime)

class PATIENT_PERSONAL(DBActivityModel):

    __tablename__ = "patient_personal"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    highest_education = Column(Integer, ForeignKey(ENUMLISTS.id))
    height = Column(Float)
    weight = Column(Integer)

class RESULT_TYPES(DBActivityModel):

    __tablename__ = "result_types"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    code = Column(VARCHAR(10), nullable=False)
    description = Column(VARCHAR(256))

class RESULT_MOMENTS(DBActivityModel):

    __tablename__ = "result_moments"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    code = Column(VARCHAR(10), nullable=False)
    description = Column(VARCHAR(256))

class PATIENT_RESULTS(DBActivityModel):

    __tablename__ = "patient_results"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False)
    result_type = Column(Integer, ForeignKey(RESULT_TYPES.id), nullable=False)
    result_moment = Column(Integer, ForeignKey(RESULT_MOMENTS.id), nullable=False)
    int_value = Column(Integer)
    float_value = Column(Float)
    string_value = Column(VARCHAR(256))    

class THRESHOLD_TYPES(DBActivityModel):

    __tablename__ = "thresholds_types"

    id = Column(VARCHAR(5), nullable=False, primary_key=True)
    description = Column(VARCHAR(256), nullable=False)

class THRESHOLDS(DBActivityModel):

    __tablename__ = "thresholds"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False)
    pre_or_post = Column(VARCHAR(4))
    year = Column(Integer)
    week = Column(Integer)
    weekday = Column(Integer)
    threshold_type = Column(VARCHAR(5),ForeignKey(THRESHOLD_TYPES.id), nullable=False)
    threshold = Column(Float, nullable=False, default=0)

class CLINICAL_PICTURE(DBActivityModel):
    
    __tablename__ = "clinical_picture"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    work_capacity_preop = Column(Integer)
    number_of_days_of_work_preop = Column(Integer)
    number_of_problematic_days = Column(Integer)
    amount_of_work_on_problematic_days = Column(Integer)
    problem_unpaid_workdays = Column(Integer)
    preop_work_capacity = Column(Float)

class RFE_RESULTS(DBActivityModel):

    __tablename__ = "rfe_results"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    rundate = Column(DateTime, default=datetime.now)
    estimator = Column(VARCHAR(10), nullable=False)
    nr_classes = Column(Integer, nullable=False)
    nr_features = Column(Integer, nullable=False)
    nr_selected = Column(Integer, nullable=False)
    nr_runs = Column(Integer, nullable=False)
    nr_splits = Column(Integer)
    split_group = Column(Integer)

class RFE_RESULTS_FEATURES(DBActivityModel):

    __tablename__ = "rfe_results_features"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    rfe_results_id = Column(Integer, ForeignKey(RFE_RESULTS.id))
    feature = Column(VARCHAR(50), nullable=False)
    ranking = Column(Integer, nullable=False)
    coefficient = Column(Float)
    importance = Column(Float)

class RFE_SCORING_TYPES(DBActivityModel):
    __tablename__ = "rfe_scoring_types"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    name = Column(VARCHAR(40), nullable=False)

class RFE_SCORING(DBActivityModel):

    __tablename__ = "rfe_scoring"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    rfe_results_id = Column(Integer, ForeignKey(RFE_RESULTS.id))
    test_or_train_data = Column(VARCHAR(5))
    scoring_type_id = Column(Integer, ForeignKey(RFE_SCORING_TYPES.id))
    value = Column(Float, nullable=False)
    
class RTW_CLASSES(DBActivityModel):

    __tablename__ = "rtw_classes"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    class_id = Column(Integer, nullable=False)
    nr_of_classes = Column(Integer, nullable=False)
    from_week = Column(Float, nullable=False)    
    to_week = Column(Float, nullable=False)

class PATIENT_ACTIVITY(DBActivityModel):

    __tablename__ = "patient_activity"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    average_inactivity_pre = Column(Float)
    average_inactivity_post = Column(Float)
    average_steps_pre = Column(Float)
    average_steps_post = Column(Float)

class RFE_RANKING_SCORING(DBActivityModel):

    __tablename__ = "rfe_ranking_scoring"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    rundate = Column(DateTime, default=datetime.now)
    estimator = Column(VARCHAR(10), nullable=False)
    nr_classes = Column(Integer, nullable=False)
    nr_selected_max = Column(Integer, nullable=False)
    accuracy_max = Column(Float, nullable=False)
    test_or_train_data = Column(VARCHAR(5))
    ranking_score = Column(Float, nullable=False)
    
