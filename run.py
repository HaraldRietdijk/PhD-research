import sys
from database.db import create_engine_and_session, setup_db
from app import app

from steps.run_steps import do_run
from steps.run_steps_NS import do_run_NS

if __name__ == "__main__":
    schema="activity"
    force=False
    testing=False
    do_steps=None
    if (len(sys.argv)<2):
        print('Usage: python run.py mode arg arg2 arg3\n'
              '       mode: normal or teststuff\n'
              '             normal for the HFT schema and functions teststuff to run trials and test\n'
              '       arg: all, rawdata, stepX test or force\n'
              '            Use all to run all steps\n'
              '            Use stepX to run step X(1 to 9)\n'
              '            Use rawdata to load the Fitbit raw-data\n'
              '            Use test to run the test for given schema\n'
              '            Use force to reset the database completely and recreate all db-objects for the selected schema'
              '       arg2: orig ns\n'
              '            Use orig to process the original data\n'
              '            Use ns to process the NijSmellinghe data\n'
              '       arg3: (Optional) clustering type\n'
              '            vfc, ns, full for clustering on parts of the dataset, default vfc\n'
              '            or for step 10 (Optional): full, filter, lasso, shap, plot, default plot\n'
              '            Use full to generate the data for all FS methods or select a method\n'          
              '            Graphs are always reproduced\n'          
              '            Use plot (default) to only reproduce the graphs\n'          
              '       arg4: (Optional) folder names\n'
              '            Folder name for storing models, plots and logging\n'
              '       arg5: (Optional) number of clusters\n'
              '            4, 7, 8, 9 , default 4\n')

    else:
        if sys.argv[3]=='orig':
            do_steps = do_run
        else:
            do_steps = do_run_NS
        if sys.argv[2]=="force":
            force=True
        create_engine_and_session(app, schema)
        setup_db(app, force, testing, schema)
        do_steps(app)
