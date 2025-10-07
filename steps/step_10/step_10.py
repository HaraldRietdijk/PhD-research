from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run
from steps.step_10.step_10_filter_methods import get_filter_methods_scores
from steps.step_10.step_10_plot import plot_average_per_method, plot_results_per_method
from steps.step_10.step_10_save_results import save_method_results


def do_step_10(app):
    # Settings
    num_of_classes = 2
    base_folder='results/fs_shap'
    folder = base_folder
    # Logging and run administration
    start_logging(base_folder)
    run_id = 398
    # run_id = get_run_id(app,"Feature Selection", 'test', 10, 'NS')
    # # Get dataframe and fill Nan values with median values
    # dataframes = get_dataframe(app, num_of_classes)
    # # Get list of features
    # features = dataframes['X'].columns.tolist()
    # # fitted_models = classifier_selection(app, folder, dataframes, num_of_classes, run_id)
    # filter_methods_scores = get_filter_methods_scores(dataframes, features)
    # plot_results_per_method(folder, filter_methods_scores)
    # save_method_results(app, filter_methods_scores, run_id)
    plot_average_per_method(app, folder, run_id)
    # complete_run(app, run_id)