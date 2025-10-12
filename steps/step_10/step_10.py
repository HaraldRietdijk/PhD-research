import sys
from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging
from steps.step_10.step_10_filter_methods import do_filter_methods
from steps.step_10.step_10_lasso import do_lasso
from steps.step_10.step_10_shap_select import do_shap_select
from steps.step_10.step_10_plot import create_plots

def get_run_type():
    if len(sys.argv)<5:
        run_type = 'plot'
    else:
       run_type = sys.argv[4]
    if run_type=='full':
        run_all_methods = True
    else:
        run_all_methods = False
    return run_type, run_all_methods

def run_method_selection(app, run_type, run_all_methods):
    num_of_classes = 2 # Binary classification is all we are looking for.
    dataframes = get_dataframe(app, num_of_classes)
    features = dataframes['X'].columns.tolist()
    if (run_type=="filter") or run_all_methods: 
        do_filter_methods(app, dataframes, features)
    if (run_type=="lasso") or run_all_methods: 
        do_lasso(app, dataframes, features)
    if (run_type=="shap") or run_all_methods: 
        do_shap_select(app, dataframes, features)

def do_step_10(app):
    folder='results/fs_shap'
    start_logging(folder)
    run_type, run_all_methods = get_run_type()
    if run_type!='plot':
        run_method_selection(app, run_type, run_all_methods)
    create_plots(app, folder)

