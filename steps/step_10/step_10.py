from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging


def do_step_10(app):
    base_folder='results/fs_shap'
    start_logging(base_folder)
    dataframes = get_dataframe(app, 2)
    features = dataframes['X'].columns.tolist()
