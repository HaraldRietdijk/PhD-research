import pandas as pd

from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run
from steps.step_generic_code.dataframes_step_data import get_dataframe, get_cat, get_threshold
from steps.step_3.step_3_generic_collecting_results import fit_models, make_predictions
from steps.step_3.step_3_generic_storing_results import save_pickle_and_metrics, save_fitting_times

def do_step_3(app):
    folder='results/article_5/individual/VFC'
    start_logging(folder)
    run_id = get_run_id(app, 'orig run for article 5 no weekday', 'definitive', 3, 'orig')

    df_treatment_id, df_dataframe = get_dataframe(app)
    df_dataframe = get_threshold(df_dataframe)
    df_dataframe['dailysteps_cat'] = get_cat(df_dataframe, df_dataframe)

    fitting_results_all_models = fit_models(df_treatment_id, df_dataframe)
    predictions_all_models = make_predictions(df_treatment_id, df_dataframe, fitting_results_all_models)

    save_pickle_and_metrics(app, run_id, df_treatment_id, df_dataframe, folder, 
                            fitting_results_all_models, predictions_all_models)
    save_fitting_times(app, run_id, fitting_results_all_models)
    complete_run(app, run_id)