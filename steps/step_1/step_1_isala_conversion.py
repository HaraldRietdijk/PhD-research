from database.models.patient_data import PATIENT_PERSONAL
from database.models.isala_data import PATIENT_AID, PATIENT_INTAKE, PATIENT_CONDITION, PATIENT_HEALTH
from database.models.isala_data import OPERATION_DETAILS, OPERATION_COMPLICATION, OPERATION_READMISSION, ICU_DETAILS, PATIENT_DEATH

CCI_WEIGHTS = {'MI':1,'CHF':1,'PAD':1,'CVA':1,'DEM':1,'COPD':1,'CTD':1,
                   'PUD':1,'DM':1,'LIVER_DIS':1,'DM_complicated':2,'HEMIP':2,
                   'CKD':2,'TUM':2,'LEUK':2,'LYM':2, 'MOD_SEV_LIVER':3,'METASTASIS':6,'AIDS':6}

CONVERSION_MATRIX = {
    "Patient personal" : (PATIENT_PERSONAL,[("year_of_birth","year_of_birth",""),("Length","height",""),
                                            ("Weight","weight",""),("BMI","bmi","")]),
    "Patient aid" : (PATIENT_AID,[("Visual_aid","visual_aid",""),("Hearing_aid","hearing_aid",""),("Walker","walker",""),
                                  ("Walking_stick","walking_stick",""),("Crutch","crutch",""),("Wheelchair","wheelchair",""),
                                  ("Mobility_scooter","mobility_scooter",""),("Compression_stockings","compression_stockings",""),
                                  ("Walking_aid","walking_aid",""),("Home_adaptations","home_adaptations","")]),
    "Patient intake" : (PATIENT_INTAKE,[("intakedatum_appointment_start_date","appointment_start_date",""),("PG_SGA_t0","pg_sga",""),
                                        ("VSAQ","vsaq",""),("SRT_able","srt_able",""),("HADS","hads",""),
                                        ("6MWT","six_mwt","6MWT"),("number_sessions","number_sessions",""),
                                        ("number_sessions_complete","number_sessions_complete","")]),
    "Patient condition" : (PATIENT_CONDITION,[("ASA","asa",""),("Surgery_asa","surgery_asa",""),("Anemia_treatment","anemia_treatment",""),
                                              ("CTD","bindweefselziekte",""),("CVA","cerebrovasculaire_aandoening",""),
                                              ("CHF","congestief_hartfalen",""),("COPD","chronische_longziekte",""),("DEM","dementie",""),
                                              ("DM","diabetes_mellitus","DIAMEL"),("PUD","gastrointestinaal_ulcuslijden",""),
                                              ("AIDS","hiv_aids",""),("LYM","lymfe_klieren",""),("LEUK","leukemenie",""),
                                              ("TUM","maligniteit","MALIGN"),("LIVER_DIS","leverziekten","LIVER"),
                                              ("MI","myocardinfarct",""),("CKD","nierziekte",""),("HEMIP","para_hemiplegie",""),
                                              ("PAD","perifeer_vaatlijden_aneurysma","")]),
    "Patient health" : (PATIENT_HEALTH,[("Stopped_since","stopped_since",""),("Alcohol","alcohol","ALCOHOL"),("Drugs","drugs","DRUGS"),
                                        ("Two_stairs","two_stairs",""),("Dyspnea_MRC","dyspnea_mrc",""),
                                        ("Personal_care","personal_care","PERSONCARE"),("Functioning","functioning","FUNCTION"),
                                        ("KATZ","katz",""),("Corticosteroids","corticosteroids",""),
                                        ("Previous_cognitive","previous_cognitive",""),("Smoking","smoking","SMOKING")]),
    "Patient death" : (PATIENT_DEATH,[("date_death","date",""),("mortality_30days","mortality_30days","")]),
    "Operation details" : (OPERATION_DETAILS,[("Type_surgery","type_surgery","SURGERYTYP"),("LOS","los",""),("LOS_minutes","los_minutes",""),
                                              ("complicaties_dcra_form_entries_value_text","complicaties_dcra",""),
                                              ("Procedure_surgery_raw","procedure_surgery_raw","SURGERYRAW"),
                                              ("Tumor_location","tumor_location","TUMORLOC"),("Elective","elective","ELECTIVE"),
                                              ("Duration_surgery","duration_surgery",""),("Surgeon","surgeon","SURGEON"),
                                              ("Type_anesthesia","type_anesthesia","ANESTH"),("Date_surgery","date_surgery",""),
                                              ("Date_admission","date_admission",""),("Date_discharge","date_discharge",""),
                                              ("Admission_source","admission_source","ADSOURCE"),
                                              ("Discharge_source","discharge_source","DISSOURCE"),
                                              ("validation_surgery","validation_surgery","VALIDATION"),("ICU","icu",""),
                                              ("neoadjuvant","neoadjuvant","")]),
    "ICU details" : (ICU_DETAILS,[("ICU_los","los",""),("IC_start","start_date",""),("IC_end","end_date","")]),
    "Operation complication" : (OPERATION_COMPLICATION,[("Feeding_tube","feeding_tube",""),("Complications_number","complications_number",""),
                                                        ("Complication","complication",""),("Clavien_dindo","clavien_dindo","CLAVIEN"),
                                                        ("Complication_type","complication_type","")]),
    "Operation readmission" : (OPERATION_READMISSION,[("Readmission_date","date",""),("Readmission_discharge","discharge",""),
                                                      ("Readmission_los","los",""),("Reoperation_type","reoperation_type","REOPER"),
                                                      ("Readmissions_30days","readmissions",""),("Reoperation","reoperation","REOPER")])
}

BOOLEAN_FIELDS = ["Visual_aid","Hearing_aid","Walker","Walking_stick","Crutch","Wheelchair","Mobility_scooter","Compression_stockings",
                  "Walking_aid","Two_stairs","Home_adaptations","Corticosteroids","Previous_cognitive","SRT_able","Anemia_treatment",
                  "Feeding_tube","complicaties_dcra_form_entries_value_text","Complication","ICU","Readmissions_30days","Reoperation","CTD","CVA",
                  "CHF","COPD","DEM","PUD","AIDS","MI","CKD","HEMIP","PAD","LYM","LEUK","mortality_30days","neoadjuvant"]

DATE_FIELDS = ["intakedatum_appointment_start_date","Readmission_date","Readmission_discharge","Date_surgery","Date_admission",
              "Date_discharge","IC_start","IC_end","date_death"]

RESULTS_LIST = [("Hb_t0","Hb","T0"), ("Albumin","Albumin","T0"), ("Ferritin","Ferritin","T0"), ("Transferrin","Transfer","T0"),
                ("Trans_sat","Trans_sat","T0"), ("CRP","CRP","T0"), ("HbA1c","HbA1c","T0"), ("Creatinine","Creatinine","T0"),
                ("Hb_t1","Hb","T1"), ("SRT_t0","SRT","T0"), ("Handgrip_strength","Handgrip","T0"), ("5TSTS","5XCRT","T0"), 
                ("Second_rise","Secondrise","T0"), ("6MWT_m","6MWT","T0"), ("SRT_t1","SRT","T1"), ("5TSTS_t1","5XCRT","T1"),
                ("Secondrise_t1","Secondrise","T1"), ("Handgrip_t1","Handgrip","T1")]





