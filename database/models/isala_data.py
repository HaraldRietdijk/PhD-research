from sqlalchemy import Column, ForeignKey, VARCHAR, Integer, DateTime, Float, Boolean
from database.models.model import DBActivityModel
from database.models.hft_tables import HFT_TREATMENT_T
from database.models.patient_data import ENUMLISTS
from datetime import datetime


# class PATIENT_PERSONAL(DBActivityModel):

#     __tablename__ = "patient_personal"

#     treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
#     highest_education = Column(Integer, ForeignKey(ENUMLISTS.id))
#     height = Column(Float)
#     weight = Column(Integer)
#     bmi = Column(Float)
#     year_of_birth = Column(Integer)

class PATIENT_INTAKE(DBActivityModel):

    __tablename__ = "patient_intake"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    hads = Column(Integer)
    pg_sga = Column(Integer)
    vsaq = Column(Integer)
    six_mwt = Column(Integer, ForeignKey(ENUMLISTS.id))
    srt_able = Column(Boolean)
    appointment_start_date = Column(DateTime)
    number_sessions = Column(VARCHAR(100))
    number_sessions_complete = Column(VARCHAR(20))

class PATIENT_HEALTH(DBActivityModel):

    __tablename__ = "patient_health"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    corticosteroids = Column(Boolean)
    dyspnea_mrc = Column(Integer)
    stopped_since = Column(Integer)
    two_stairs = Column(Boolean)
    alcohol = Column(Integer, ForeignKey(ENUMLISTS.id))
    drugs = Column(Integer, ForeignKey(ENUMLISTS.id))
    smoking = Column(Integer, ForeignKey(ENUMLISTS.id))
    functioning = Column(Integer, ForeignKey(ENUMLISTS.id))
    personal_care = Column(Integer, ForeignKey(ENUMLISTS.id))
    katz = Column(Integer)
    previous_cognitive = Column(Boolean)

class PATIENT_CONDITION(DBActivityModel):

    __tablename__ = "patient_condition"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    anemia_treatment = Column(Boolean)
    bindweefselziekte = Column(Boolean)
    cerebrovasculaire_aandoening = Column(Boolean)
    chronische_longziekte = Column(Boolean)
    congestief_hartfalen = Column(Boolean)
    dementie = Column(Boolean)
    diabetes_mellitus = Column(Integer, ForeignKey(ENUMLISTS.id))
    gastrointestinaal_ulcuslijden = Column(Boolean)
    hiv_aids = Column(Boolean)
    leverziekten = Column(Integer, ForeignKey(ENUMLISTS.id))
    maligniteit = Column(Integer, ForeignKey(ENUMLISTS.id))
    myocardinfarct = Column(Boolean)
    nierziekte = Column(Boolean)
    para_hemiplegie = Column(Boolean)
    perifeer_vaatlijden_aneurysma = Column(Boolean)
    lymfe_klieren = Column(Boolean)
    leukemenie = Column(Boolean)
    asa = Column(Integer)
    surgery_asa = Column(Integer)

class PATIENT_AID(DBActivityModel):

    __tablename__ = "patient_aid"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    compression_stockings = Column(Boolean)
    crutch = Column(Boolean)
    hearing_aid = Column(Boolean)
    mobility_scooter = Column(Boolean)
    visual_aid = Column(Boolean)
    walker = Column(Boolean)
    walking_aid = Column(Boolean)
    walking_stick = Column(Boolean)
    wheelchair = Column(Boolean)
    home_adaptations = Column(Boolean)

# class SURGEON(DBActivityModel):

#     __tablename__ = "surgeon"
#     surgeon_id = Column(Integer, nullable=False, primary_key=True)
#     surgeon_name = Column(VARCHAR(256))

# class SURGERY_TYPE(DBActivityModel):

#     __tablename__ = "surgery_type"
#     surgery_id = Column(Integer, nullable=False, primary_key=True)
#     surgery_type = Column(VARCHAR(40))

class OPERATION_DETAILS(DBActivityModel):

    __tablename__ = "operation_details"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    date_admission = Column(DateTime)
    date_discharge = Column(DateTime)
    date_surgery = Column(DateTime)
    duration_surgery = Column(Integer)
    elective = Column(Integer, ForeignKey(ENUMLISTS.id))
    procedure_surgery_raw = Column(VARCHAR(256))
    tumor_location = Column(Integer, ForeignKey(ENUMLISTS.id))
    type_anesthesia = Column(Integer, ForeignKey(ENUMLISTS.id))
    type_surgery_code = Column(Integer)
    type_surgery = Column(Integer, ForeignKey(ENUMLISTS.id))
    neoadjuvant = Column(Boolean)
    complicaties_dcra = Column(Boolean)
    admission_source = Column(Integer, ForeignKey(ENUMLISTS.id))
    discharge_source = Column(Integer, ForeignKey(ENUMLISTS.id))
    los = Column(Integer)
    los_minutes = Column(Float)
    icu = Column(Boolean)
    validation_surgery = Column(Integer, ForeignKey(ENUMLISTS.id))
    surgeon = Column(Integer, ForeignKey(ENUMLISTS.id))

class ICU_DETAILS(DBActivityModel):

    __tablename__ = "icu_details"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    los = Column(Integer)
    end_date = Column(DateTime)
    start_date = Column(DateTime)

class OPERATION_COMPLICATION(DBActivityModel):

    __tablename__ = "operation_complication"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    complications_number = Column(Integer)
    feeding_tube = Column(Boolean)
    clavien_dindo = Column(Integer, ForeignKey(ENUMLISTS.id))
    complication = Column(Boolean)
    complication_type = Column(VARCHAR(256))

class OPERATION_READMISSION(DBActivityModel):

    __tablename__ = "operation_readmission"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    date = Column(DateTime)
    discharge = Column(DateTime)
    los = Column(Integer)
    readmissions = Column(Boolean)
    reoperation = Column(Integer, ForeignKey(ENUMLISTS.id))
    reoperation_type = Column(Integer, ForeignKey(ENUMLISTS.id))

class PATIENT_DEATH(DBActivityModel):

    __tablename__ = "patient_status"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    date = Column(DateTime)
    mortality_30days = Column(Boolean)

class PATIENT_DERIVED(DBActivityModel):

    __tablename__ = "patient_derived"

    treatment_id = Column(Integer, ForeignKey(HFT_TREATMENT_T.treatment_id), nullable=False, primary_key=True)
    cci = Column(Integer)
    los_threshold = Column(Integer)
    los_prolonged = Column(Boolean)
    non_textbook_1 = Column(Boolean)
    non_textbook_2 = Column(Boolean)
    use_for_model = Column(Boolean)
