from sqlalchemy import Column, Integer, DateTime, Float, VARCHAR, ForeignKey
from database.models.model import DBActivityModel
from database.models.hft_tables import HFT_TREATMENT_T

class FitbitData(DBActivityModel):
    """
    This table contains data that is obtained directly from Fitbit devices. The data of this table is stored per minute
    and should be converted to the hour_measurement table, which is easier to use.
    """
    __tablename__ = "fitbit_data"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    treatment_id = Column(Integer, nullable=False)
    fitbit_id = Column(Integer, nullable=False)
    datetime = Column(DateTime, nullable=False)
    calories = Column(Float(32, 2), default=0)
    mets = Column(Integer, nullable=False)
    level = Column(Integer, nullable=False)
    steps = Column(Integer, default=0)
    distance = Column(Float(32, 2), default=0)

    def to_dict(self):
        return dict(
            id=self.id,
            treatment_id = self.treatment_id,
            fitbit_id = self.fitbit_id,
            datetime = self.datetime,
            calories = self.calories,
            mets = self.mets,
            level = self.level,
            steps = self.steps,
            distance = self.distance
        )
    
class AcceleroID(DBActivityModel):
    __tablename__ = "accelero_ids"

    patient_id = Column(VARCHAR(6), nullable=False, primary_key=True)
    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False)

class AcceleroData(DBActivityModel):
    __tablename__ = "accelero_data"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    patient_id = Column(VARCHAR(6), ForeignKey(AcceleroID.patient_id), nullable=False)
    datetime = Column(DateTime, nullable=False)
    score = Column(Float(32, 2), nullable=False)
    mets = Column(Float(32,4), nullable=False)
    steps = Column(Integer, default=0)
