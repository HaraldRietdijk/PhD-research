import pandas as pd
from sqlalchemy import text, and_, or_, not_, func


from database.models.hft_views import SumSteps
from steps.step_generic_code.dataframes_step_data import get_cat, get_trainsets
from steps.step_2.step_2_generic import get_results, plot_graph, save_results, save_times
from steps.step_generic_code.dataframes_step_data import get_dataframe
from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run

def get_threshold_df(df_dataframe):
    average = [df_dataframe.loc[(df_dataframe.hour==18)].sum_steps_hour.mean()]
    result = pd.DataFrame([row for row in average], columns=['threshold'])
    return result

def do_step_2(app):
    folder='results/article_5'
    start_logging(folder)
    run_id = get_run_id(app, 'orig run for article 5 no weekday', 'definitive', 2, 'orig')

    
    _, df_dataframe = get_dataframe(app)
    df_threshold = get_threshold_df(df_dataframe)

    df_dataframe['dailysteps_cat'] = get_cat(df_dataframe, df_threshold)

    test_size = 0.3
    X_train_s,y_train_s,X_train,y_train = get_trainsets(df_dataframe,test_size)

    results, names, times = get_results(X_train_s,y_train_s,X_train,y_train,"f1")
    save_results(app, run_id,results,names,"f1")
    save_times(app, run_id, times, 1000)
    plot_graph(results, names, 'Algorithm F1-score Comparison', 'F1_General', folder)
    
    results, names, times = get_results(X_train_s,y_train_s,X_train,y_train,"accuracy")
    save_results(app, run_id, results, names, "accuracy")
    save_times(app, run_id, times, 1001)
    plot_graph(results, names, 'Algorithm Accuracy Comparison', 'Accuracy_General', folder)

    complete_run(app, run_id)