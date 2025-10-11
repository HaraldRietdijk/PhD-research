import os
import sys
import logging

from database.models.hft_tables import HFT_RUN_T

def pickle_destination(root,leaf):
    dest = os.path.join(root,leaf) 
    if not os.path.exists(dest): 
        os.makedirs(dest) 
    return dest 

def convert_list(list):
    string_value='['
    first=True
    for value in list:
        if first:
            first=False
        else:
            string_value+=', '
        string_value+=str(value)
    string_value+=']'
    return string_value

def check_folder(folder):
    if not os.path.exists(folder): 
        os.makedirs(folder) 

def start_logging(folder='logging_folder'):
    if len(sys.argv)>=6:
        folder = 'results/' + sys.argv[5]
    check_folder(folder) 
    logging_file=folder+"/logging.txt"
    logging.basicConfig(filename=logging_file,level=logging.ERROR)
    logging.captureWarnings(True)

def get_run_id(app, description, run_type, run_step, data_set):
    run = HFT_RUN_T(description = description,
                    run_type = run_type,
                    run_step = run_step,
                    data_set = data_set,
                    run_completed = False)
    app.session.add(run)
    app.session.commit()
    return run.run_id

def complete_run(app, run_id):
    run = app.session.query(HFT_RUN_T).filter(HFT_RUN_T.run_id==run_id).first()
    run.run_completed = True
    app.session.add(run)
    app.session.commit()