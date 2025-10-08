from steps.step_10.step_10_lasso import get_lasso_scores
from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run
from steps.step_10.step_10_filter_methods import get_filter_methods_scores
from steps.step_10.step_10_plot import plot_average_per_method, plot_results_per_method
from steps.step_10.step_10_save_results import save_method_results

def do_filter_methods(app, dataframes, features, folder):
    # run_id=398 # use for reprinting specific run
    for i in range(5):
        print('Starting filter run: ',str(i+1))
        run_id = get_run_id(app,"Feature Selection filter", 'test', 10, 'NS')
        filter_methods_scores = get_filter_methods_scores(dataframes, features)
        plot_results_per_method(folder, filter_methods_scores, run_id)
        save_method_results(app, filter_methods_scores, run_id)
        plot_average_per_method(app, folder, run_id)
        complete_run(app, run_id)
    plot_average_per_method(app, folder)

def do_lasso(app, dataframes, features, folder):
    lasso_scores = {}
    for i in range(5):
        print('Starting LASSO run: ',str(i+1))
        run_id = get_run_id(app,"Feature Selection LASSO", 'test', 10, 'NS')
        thresholds = [0.05, 0.1, 0.15, 0.2]
        lasso_scores['LASSO'] = get_lasso_scores(dataframes, features, thresholds)
        plot_results_per_method(folder, lasso_scores, run_id, thresholds=thresholds)
        save_method_results(app, lasso_scores, run_id, thresholds=thresholds)
        plot_average_per_method(app, folder, run_id, thresholds=thresholds)
        complete_run(app, run_id)

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
