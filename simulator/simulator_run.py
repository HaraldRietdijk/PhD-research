import logging

from simulator.data_generation import data_generation
from simulator.database_functions import get_drift_settings, add_treatment_and_run_id, get_movement_patterns, store_daily_patterns, store_daily_steps
from steps.step_generic_code.general_functions import get_run_id, start_logging, complete_run
from simulator.import_settings import get_settings_input

def do_simulator_run(app):
    folder='results/simulator'
    start_logging(folder)
    inFile = "simulator/test.txt"
    movement_patterns, patterns_by_id = get_movement_patterns(app)    
    input_settings = get_settings_input(app, inFile, movement_patterns)
    if input_settings:
        run_id = get_run_id(app, 'simulator run', 'test', 10, 'orig')
        for settings in input_settings:
            settings = add_treatment_and_run_id(app, settings, run_id)
            settings = get_drift_settings(app, settings)
            settings.movement_pattern_name = patterns_by_id[settings.movement_pattern]
            daily_steps, daily_patterns = data_generation(settings, movement_patterns)
            store_daily_steps(app, daily_steps, settings)
            store_daily_patterns(app, daily_patterns, settings)
        complete_run(app, run_id)
    else:
        logging.info("No new settings found.")
