from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging

from steps.step_10.step_10_selection_classifier import classifier_selection

from steps.step_generic_code.general_variables.general_variables_all_shap import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS
from steps.step_10.step_10_generic_selection import save_pickle_and_metrics

from steps.step_10.step_10_plot import plot_results


def do_step_10(app):
    # Settings
    num_of_classes = 2
    base_folder='results/fs_shap'
    # Logging
    start_logging(base_folder)
    # Get dataframe and fill Nan values with meadium values
    dataframes = get_dataframe(app, num_of_classes)
    
    # Get list of features
    features = dataframes['X'].columns.tolist()

    ### Train models and save results
    folder = base_folder + '/classifiers'
    fitted_models = classifier_selection(app, folder, dataframes, num_of_classes)
    plot_results(app, folder, num_of_classes)