from steps.step_10.step_10_lasso import do_lasso
from steps.step_10.step_10_shap_select import do_shap_select
from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging
from steps.step_10.step_10_filter_methods import do_filter_methods
from steps.step_10.step_10_plot import plot_accuracy_for_all_methods_per_features, plot_accuracy_for_all_methods_per_model, plot_metrics_over_all_models_per_method, plot_metrics_per_model_per_method

def do_step_10(app):
    num_of_classes = 2
    base_folder='results/fs_shap'
    folder = base_folder
    start_logging(base_folder)
    dataframes = get_dataframe(app, num_of_classes)
    features = dataframes['X'].columns.tolist()
    do_filter_methods(app, dataframes, features)
    do_lasso(app, dataframes, features)
    do_shap_select(app, dataframes, features)
    plot_metrics_over_all_models_per_method(app, folder) # 4 graphs (4 methods) with each four lines, one line per metric, taking the average metric over all models
    plot_accuracy_for_all_methods_per_features(app,folder) # one graph with four lines, one line per method, average over all models
    plot_metrics_per_model_per_method(app,folder) # 24 graphs (3*8 models) with four lines, one per metric
    plot_accuracy_for_all_methods_per_model(app,folder) # 24 graphs (3*8 models) with four lines, one per metric

