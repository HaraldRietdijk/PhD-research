from sqlalchemy import Column, ForeignKey, VARCHAR, Integer, Float, Boolean
from database.models.model import DBActivityModel
from database.models.hft_tables import HFT_RUN_T

class SELECTION_METHOD(DBActivityModel):

    __tablename__ = "fs_selection_method"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    type = Column(VARCHAR(10), nullable=False)
    name = Column(VARCHAR(10), nullable=False)
    use_for_ordering = Column(Boolean)
    create_plot = Column(Boolean)

class METHOD_RESULTS(DBActivityModel):

    __tablename__ = "fs_method_results"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    method_id = Column(Integer, ForeignKey(SELECTION_METHOD.id), nullable=False)
    run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=False)
    model = Column(VARCHAR(10), nullable=False)
    nr_features = Column(Integer, nullable=False)
    threshold = Column(Float, default=0)
    accuracy = Column(Float)
    f1_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)

class METHOD_RESULTS_FEATURES(DBActivityModel):

    __tablename__ = "fs_method_results_features"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    result_id = Column(Integer, ForeignKey(METHOD_RESULTS.id), nullable=False)
    feature = Column(VARCHAR(50), nullable=False)
    coefficient = Column(Float)

class METHOD_MODEL_AVERAGE(DBActivityModel):

    __tablename__ = "fs_method_model_average"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    method_id = Column(Integer, ForeignKey(SELECTION_METHOD.id), nullable=False)
    model = Column(VARCHAR(10))
    data_set = Column(VARCHAR(10))
    nr_features = Column(Integer)
    accuracy = Column(Float)
    f1_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)

class METHOD_MODEL_SCORING(DBActivityModel):

    __tablename__ = "fs_method_model_scoring"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    method_id = Column(Integer, ForeignKey(SELECTION_METHOD.id), nullable=False)
    model = Column(VARCHAR(10))
    data_set = Column(VARCHAR(10))
    max_accuracy = Column(Float)
    nr_features = Column(Integer)
    s_ordering = Column(Float)
    ordering_loss = Column(Float)

class FEATURES_SCORING(DBActivityModel):

    __tablename__ = "fs_features_scoring"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    method_model_id = Column(Integer, ForeignKey(METHOD_MODEL_SCORING.id), nullable=False)
    feature =  Column(VARCHAR(50), nullable=False)
    ranking = Column(Integer)
    accuracy_delta = Column(Float)
    model_accuracy = Column(Float)
    s_ordering_contribution = Column(Float)

class FEATURES_RANKING_GROUP(DBActivityModel):

    __tablename__ = "fs_features_ranking_group"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    method_id = Column(Integer, ForeignKey(SELECTION_METHOD.id), nullable=False)
    data_set = Column(VARCHAR(10))
    s_ordering = Column(Float)
    max_accuracy = Column(Float)
class FEATURES_RANKING(DBActivityModel):

    __tablename__ = "fs_features_ranking"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    group_id = Column(Integer, ForeignKey(FEATURES_RANKING_GROUP.id), nullable=False)
    feature =  Column(VARCHAR(50), nullable=False)
    ranking = Column(Integer)
    ranking_score = Column(Float)
    s_ordering_contribution = Column(Float)
    accuracy_delta = Column(Float)
    model_accuracy = Column(Float)