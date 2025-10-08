from sqlalchemy import Column, ForeignKey, VARCHAR, Integer, Float
from database.models.model import DBActivityModel
from database.models.hft_tables import HFT_RUN_T

class SELECTION_METHOD(DBActivityModel):

    __tablename__ = "fs_selection_method"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    type = Column(VARCHAR(10), nullable=False)
    name = Column(VARCHAR(10), nullable=False)

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
