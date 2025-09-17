import pandas as pd

from database.models.hft_tables import HFT_TREATMENT_T

def update_treatment_data(app, file_name):
    print("Updating treatment table")
    df_from_treatment_file = pd.read_csv(file_name)
    rows = [tuple(x) for x in df_from_treatment_file.values]
    for row in rows:
        treatment_id = row[3]
        treatment = app.session.query(HFT_TREATMENT_T).filter_by(treatment_id=treatment_id).first()
        if (treatment): 
            treatment.gender=row[2]
            treatment.research_group=row[1]
            app.session.add(treatment)
            app.session.commit()
        else:
            print('treatment_id not found:',treatment_id)

def add_treatment_data(app):
#   Adding additional treatment data: gender and research group
    file_name = './data_origin/source/Treatment_data.csv'
    update_treatment_data(app, file_name)
