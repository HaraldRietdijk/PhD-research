from steps.step_10.step_10_lasso import do_lasso
from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging
from steps.step_10.step_10_filter_methods import do_filter_methods
from steps.step_10.step_10_plot import plot_average_per_method

def do_step_10(app):
    # Settings
    num_of_classes = 2
    base_folder='results/fs_shap'
    folder = base_folder
    # Logging and run administration
    start_logging(base_folder)
    # Get dataframe and fill Nan values with median values
    dataframes = get_dataframe(app, num_of_classes)
    # Get list of features
    features = dataframes['X'].columns.tolist()
    do_filter_methods(app, dataframes, features, folder)
    do_lasso(app, dataframes, features, folder)
    plot_average_per_method(app, folder)
