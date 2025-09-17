from sqlalchemy import Column, Index, CheckConstraint, ForeignKey, VARCHAR, Integer, DateTime, Float, Boolean
from database.models.model import DBActivityModel
from datetime import datetime

class HFT_RUN_T(DBActivityModel):
    __tablename__ = "hft_run_t"

    run_id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    rundate = Column(DateTime, default=datetime.now)
    description = Column(VARCHAR(150))
    run_type = Column(VARCHAR(45))
    run_step = Column(VARCHAR(10))
    run_completed = Column(Boolean, default=False)
    data_set = Column(VARCHAR(45))

class HFT_ALGORITHM_T(DBActivityModel):
    __tablename__ = "hft_algorithm_t"

    name = Column(VARCHAR(150), nullable=False, primary_key=True)
    description = Column(VARCHAR(45))

class HFT_TREATMENT_T(DBActivityModel):
    __tablename__ = "hft_treatment_t"

    treatment_id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    age = Column(Integer)
    gender = Column(VARCHAR(1))
    research_group = Column(Integer)

    __table_args__ = (CheckConstraint(gender.in_(['f','m','u'])), )

    def __repr__(self):
        return f"HFT_TREATMENT_T({self.treatment_id!r}, {self.age!r}, {self.gender!r},{self.research_group!r})"

class HFT_MODEL_T(DBActivityModel):
    __tablename__ = "hft_model_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    name = Column(VARCHAR(50), nullable=False)
    algorithm = Column(VARCHAR(40), nullable=False)
    usedate = Column(DateTime, default=datetime.now)
    destination = Column(VARCHAR(50))
    random_seed = Column(Integer, nullable=False, default=-1)

class HFT_DATA_T(DBActivityModel):
    __tablename__ = "hft_data_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    steps = Column(Integer)
    mdate = Column(DateTime)
    year = Column(Integer)
    week = Column(Integer)
    weekday = Column(Integer)
    hour = Column(Integer)
    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False)

    __table_args__ = (Index('INDEX1', "year", "week", "weekday", "hour"), Index('INDEX2', "year"),
                       Index('INDEX3', "week"), Index('INDEX4', "hour"), Index('TREATMENT_ID_IDX', "treatment_id"), )

    def __repr__(self):
        return f"HFT_DATA_T({self.id!r}, {self.treatment_id!r}, {self.steps!r},{self.mdate!r}, \
            {self.week!r}, {self.weekday!r},{self.year!r}, {self.hour!r})"

class HFT_METRICS_T(DBActivityModel):
    __tablename__ = "hft_metrics_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    f1_score = Column(Float)
    true_negative = Column(Integer)
    true_positive = Column(Integer)
    false_negative = Column(Integer)
    false_positive = Column(Integer)
    rundate = Column(DateTime, default=datetime.now)
    startdate = Column(DateTime)
    enddate = Column(DateTime)
    accuracy = Column(Float)
    hft_model_id = Column(Integer, ForeignKey(HFT_MODEL_T.id), nullable=False)
    hft_treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)
    threshold = Column(Integer)
    number_of_observations = Column(Integer)
    weekday = Column(Integer)

class HFT_ESTIMATOR_METRICS_T(DBActivityModel):
    __tablename__ = "hft_estimator_metrics_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    rundate = Column(DateTime, default=datetime.now)
    hft_model_id = Column(Integer, ForeignKey(HFT_MODEL_T.id), nullable=False)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)
    nr_of_classes = Column(Integer, default=0)

class HFT_METRICS_TYPES_T(DBActivityModel):
    __tablename__ = "hft_metrics_types_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    name = Column(VARCHAR(40), nullable=False)

class HFT_ESTIMATOR_METRICS_RESULTS_T(DBActivityModel):
    __tablename__ = "hft_estimator_metrics_results_t"

    estimator_metrics_id = Column(Integer, ForeignKey(HFT_ESTIMATOR_METRICS_T.id), primary_key=True)
    metrics_type_id = Column(Integer, ForeignKey(HFT_METRICS_TYPES_T.id), primary_key=True)
    test_or_train = Column(VARCHAR(5), primary_key=True)
    value = Column(Float, nullable=False)

class HFT_PARAMETERS_T(DBActivityModel):
    __tablename__ = "hft_parameters_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    name = Column(VARCHAR(150), nullable=False)
    hft_algorithm_t_name = Column(VARCHAR(150), ForeignKey(HFT_ALGORITHM_T.name), nullable=False)

class HFT_MODEL_PARAMETERS_T(DBActivityModel):
    __tablename__ = "hft_model_parameters_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    value = Column(VARCHAR(2000))
    hft_model_id = Column(Integer, ForeignKey(HFT_MODEL_T.id), nullable=False)
    hft_parameters_t_id = Column(Integer, ForeignKey(HFT_PARAMETERS_T.id), nullable=False)
    treatment_id = Column(Integer)

class HFT_FITTING_PARAMETERS_T(DBActivityModel):
    __tablename__ = "hft_fitting_parameters_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    hft_algorithm_t_name = Column(VARCHAR(150), ForeignKey(HFT_ALGORITHM_T.name), nullable=False)
    hft_parameters_t_id = Column(Integer, ForeignKey(HFT_PARAMETERS_T.id), nullable=False)
    value = Column(VARCHAR(2000))
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)

class HFT_PREDICTION_T(DBActivityModel):
    __tablename__ = "hft_prediction_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    steps_cat = Column(Integer, nullable=False)
    prediction_cat = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    hft_data_t_id = Column(Integer, ForeignKey(HFT_DATA_T.id))
    hft_model_t_id = Column(Integer, ForeignKey(HFT_MODEL_T.id), nullable=False)

    __table_args__ = (Index('INDEX5', "hft_data_t_id", "hft_model_t_id"),  )

class HFT_FITTING_TIME_T(DBActivityModel):
    __tablename__ = "hft_fitting_time_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    hft_treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)
    algorithm = Column(VARCHAR(40), nullable=False)
    fitting_time_sec = Column(Integer)
    calculation_date = Column(DateTime, default=datetime.now)
    random_seed = Column(Integer, nullable=False, default=-1)

class HFT_CHARACTERISTIC_CATEGORY_T(DBActivityModel):
    __tablename__ = "hft_characteristic_category_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    category_name = Column(VARCHAR(256), nullable=False)

class HFT_DATA_CHARACTERISTIC_T(DBActivityModel):
    __tablename__ = "hft_data_characteristic_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    characteristic_name = Column(VARCHAR(256), nullable=False)
    description = Column(VARCHAR(2000))
    category_id = Column(Integer, ForeignKey(HFT_CHARACTERISTIC_CATEGORY_T.id), nullable=False)

class HFT_DATASET_CHARACTERISTIC_T(DBActivityModel):
    __tablename__ = "hft_dataset_characteristic_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    hft_treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False)
    characteristic_id = Column(Integer, ForeignKey(HFT_DATA_CHARACTERISTIC_T.id), nullable=False)
    value = Column(Float)
    error = Column(Float)
    calculation_time_sec = Column(Integer)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)
    random_seed = Column(Integer, nullable=0, default=-1)

class CLASSIFICATION_METRICS_T(DBActivityModel):
    __tablename__ = "classification_metrics_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    f1_score = Column(Float)
    true_negative = Column(Integer)
    true_positive = Column(Integer)
    false_negative = Column(Integer)
    false_positive = Column(Integer)
    rundate = Column(DateTime, default=datetime.now)
    accuracy = Column(Float)
    model_id = Column(Integer, ForeignKey(HFT_MODEL_T.id), nullable=False)
    number_of_observations = Column(Integer)

class HFT_METRICS_GENERAL_MODELT(DBActivityModel):
    __tablename__ = "hft_metrics_general_model_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    algorithm = Column(VARCHAR(150), ForeignKey(HFT_ALGORITHM_T.name), nullable=False)
    scoring = Column(VARCHAR(10), nullable=False)
    rundate = Column(DateTime, default=datetime.now)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)
    score = Column(Float)

class CLUSTER_FEATURES_GROUP_T(DBActivityModel):
    __tablename__ = "cluster_features_group_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    nr_of_features = Column(Integer)
    algorithm = Column(VARCHAR(40), nullable=False)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)

class CLUSTER_FEATURES_T(DBActivityModel):
    __tablename__ = "cluster_features_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    hft_feature_group_id = Column(Integer, ForeignKey(CLUSTER_FEATURES_GROUP_T.id), nullable=True)
    hft_parameters_t_id = Column(Integer, ForeignKey(HFT_PARAMETERS_T.id), nullable=False)

class CLUSTER_GENERATED_T(DBActivityModel):
    __tablename__ = "cluster_generated_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    cluster_group = Column(Integer, nullable=False)
    original_group = Column(VARCHAR(40), nullable=False)
    cluster_size = Column(Integer)
    hft_feature_group = Column(Integer, ForeignKey(CLUSTER_FEATURES_GROUP_T.id), nullable=True)

class CLUSTER_METRICS_T(DBActivityModel):
    __tablename__ = "cluster_metric_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    hft_feature_group_id = Column(Integer, ForeignKey(CLUSTER_FEATURES_GROUP_T.id), nullable=True)
    rand_index = Column(Float)
    adjusted_rand_index = Column(Float)
    fowlkes_mallows_score = Column(Float) 
    homogeneity = Column(Float)
    completeness = Column(Float)
    v_measure = Column(Float)
    mutual_info = Column(Float)
    adjusted_mutual_info = Column(Float)

class CLUSTER_TREATMENT_T(DBActivityModel):
    __tablename__ = "cluster_treatment_t"

    hft_treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    cluster_generated = Column(Integer, ForeignKey(CLUSTER_GENERATED_T.id), nullable=False, primary_key=True)

class CLUSTER_RUN_INFO_T(DBActivityModel):
    __tablename__ = "cluster_run_info_t"

    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True, primary_key=True)
    nr_of_clusters = Column(Integer)
    optimal_feature_group = Column(Integer, ForeignKey(CLUSTER_FEATURES_GROUP_T.id), nullable=True)
    selected = Column(Boolean)

class CLUSTERING_SELECTED_T(DBActivityModel):
    __tablename__ = "clustering_selected_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    run_id_origin = Column(Integer, ForeignKey(CLUSTER_RUN_INFO_T.hft_run_id), nullable=True)
    hft_feature_group = Column(Integer, ForeignKey(CLUSTER_FEATURES_GROUP_T.id), nullable=True)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)

class FEATURE_GROUP_CLUSTER_METRICS(DBActivityModel):
    __tablename__ = "feature_group_cluster_metrics_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    hft_feature_group = Column(Integer, ForeignKey(CLUSTER_FEATURES_GROUP_T.id), nullable=True)
    cluster_group = Column(Integer, nullable=False)
    hft_model_id = Column(Integer, ForeignKey(HFT_MODEL_T.id), nullable=False)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=False)
    f1_score = Column(Float)
    true_negative = Column(Integer)
    true_positive = Column(Integer)
    false_negative = Column(Integer)
    false_positive = Column(Integer)
    accuracy = Column(Float)
    threshold = Column(Integer)
    number_of_observations = Column(Integer)

class FEATURE_GROUP_CLUSTER_TIMES(DBActivityModel):
    __tablename__ = "feature_group_cluster_times_t"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    hft_feature_group = Column(Integer, ForeignKey(CLUSTER_FEATURES_GROUP_T.id), nullable=True)
    cluster_group = Column(Integer, nullable=False)
    hft_run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=False)
    algorithm = Column(VARCHAR(40), nullable=False)
    fitting_time_sec = Column(Integer)
