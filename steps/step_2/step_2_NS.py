from sqlalchemy import func
import pandas as pd

from database.models.patient_data import THRESHOLDS
from steps.step_generic_code.dataframes_step_data import get_cat, get_trainsets
from steps.step_2.step_2_generic import get_results, plot_graph, save_results, save_times
from steps.step_generic_code.dataframes_step_data import get_dataframe_ns
from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run

def get_threshold(app, type):
    print("Step 2: Getting threshold")
    result = app.session.query(func.avg(THRESHOLDS.threshold)).filter(THRESHOLDS.threshold_type==type)
    return pd.DataFrame([row for row in result],columns=['threshold'])

def do_step_2_NS(app):
    folder='results/article_5'
    start_logging(folder)
    run_id = get_run_id(app, 'orig run for article 5 no weekday', 'definitive', 2, 'NS')

    _, df_dataframe = get_dataframe_ns(app)
    df_threshold = get_threshold(app,'LIN_W')
    df_dataframe['dailysteps_cat']=get_cat(df_dataframe, df_threshold)

    test_size = 1-(30000/df_dataframe.shape[0])
    X_train_s,y_train_s,X_train,y_train = get_trainsets(df_dataframe, test_size)

    results, names, times = get_results(X_train_s,y_train_s,X_train,y_train,"f1")
    save_results(app, run_id, results,names, "f1")
    save_times(app, run_id, times, 1002)
    plot_graph(results, names, 'Algorithm F1-score Comparison', 'F1_General_NS', folder)
    
    results, names, times = get_results(X_train_s,y_train_s,X_train,y_train,"accuracy")
    save_results(app, run_id, results, names, "accuracy")
    save_times(app, run_id, times, 1003)
    plot_graph(results, names, 'Algorithm Accuracy Comparison', 'Accuracy_General_NS', folder)

    complete_run(app, run_id)