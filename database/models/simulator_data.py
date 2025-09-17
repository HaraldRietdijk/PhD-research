from sqlalchemy import CheckConstraint, Column, ForeignKey, VARCHAR, Integer, DateTime, Float, Boolean
from database.models.model import DBActivityModel
from database.models.hft_tables import HFT_RUN_T, HFT_TREATMENT_T
from datetime import datetime

class MOVEMENT_PATTERN(DBActivityModel):

    __tablename__ = "movement_pattern"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    name = Column(VARCHAR(20), nullable=False)
    description = Column(VARCHAR(256))
    step_adjustment = Column(Integer, nullable=True)
    adjustment_type = Column(VARCHAR(1), nullable=True)

class PEAK_WIDTH(DBActivityModel):

    __tablename__ = "peak_width"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    width_from = Column(Integer, nullable=False)
    width_to = Column(Integer, nullable=False)

class PATTERN_PEAK(DBActivityModel):

    __tablename__ = "pattern_peak"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    pattern_id = Column(Integer, ForeignKey(MOVEMENT_PATTERN.id))
    hour_before = Column(Integer, nullable=False)
    hour_from = Column(Integer, nullable=False)
    hour_to = Column(Integer, nullable=False)
    width_id = Column(Integer, ForeignKey(PEAK_WIDTH.id))
    
class SETTINGS_INPUT(DBActivityModel):

    __tablename__ = "settings_input"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=True)
    run_id = Column(Integer, ForeignKey(HFT_RUN_T.run_id), nullable=True)
    age  = Column(Integer, nullable=True)
    gender = Column(VARCHAR(1),nullable=True)
    research_group = Column(Integer, nullable=True)
    start_date = Column(DateTime, default=datetime.now)
    max_simulation_time = Column(Integer, nullable=False)
    get_threshold = Column(Boolean)
    movement_pattern = Column(Integer, ForeignKey(MOVEMENT_PATTERN.id), nullable=False)
    movement_intensity = Column(VARCHAR(8))

    __table_args__ = (CheckConstraint(gender.in_(['f','m','u'])), CheckConstraint(movement_intensity.in_(['high','average','low'])), )

class SETTINGS_DRIFT(DBActivityModel):

    __tablename__ = "settings_drift"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    settings_id = Column(Integer, ForeignKey(SETTINGS_INPUT.id), nullable=False)
    boundary = Column(Integer, nullable=False)
    percentage = Column(Integer, nullable=False)

class SETTINGS_MOTIVATION(DBActivityModel):

    __tablename__ = "settings_motivation"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    settings_id = Column(Integer, ForeignKey(SETTINGS_INPUT.id), nullable=False)
    hour = Column(Integer, nullable=False)
    effect = Column(VARCHAR(10))
    step_adjustment = Column(VARCHAR(10))

    __table_args__ = (CheckConstraint(effect.in_(['positive','negative','neutral','random'])), )

class MOVEMENT_PATTERN_USED(DBActivityModel):

    __tablename__ = "movement_pattern_used"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    settings_id = Column(Integer, ForeignKey(SETTINGS_INPUT.id), nullable=False)
    pattern_id = Column(Integer, ForeignKey(MOVEMENT_PATTERN.id))
    mdate = Column(DateTime)

