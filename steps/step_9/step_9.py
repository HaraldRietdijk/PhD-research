from steps.step_generic_code.general_functions import start_logging
from steps.step_generic_code.dataframe_knee_operations import get_dataframe

from steps.step_9.step_9_boxplots import plot_boxplots
from steps.step_9.step_9_collect_stats import get_dataset_measures
from steps.step_9.step_9_scatterplots import plot_scatterplots
from steps.step_9.step_9_OLS import datasets_info

def do_step_9(app):
    folder='results/final_review'
    start_logging(folder)
    dataframes = get_dataframe(app, 0)
    features = dataframes['X'].columns.tolist()
    # get_dataset_measures(dataframes, features, folder)
    # plot_boxplots(dataframes, features, folder)
    plot_scatterplots(dataframes, features, folder)
    # datasets_info(dataframes, folder)
    # mean_RTW = dataframes['Y_train'].mean()
    # std_RTW = dataframes['Y_train'].std(ddof=0)
    # z = (61.71-mean_RTW)/std_RTW
    # print(z, mean_RTW, std_RTW)
    # mean_RTW = dataframes['Y'].mean()
    # std_RTW = dataframes['Y'].std(ddof=0)
    # z = (61.71-mean_RTW)/std_RTW
    # print(z, mean_RTW, std_RTW)