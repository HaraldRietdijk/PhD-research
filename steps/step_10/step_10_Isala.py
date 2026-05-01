from steps.step_10.step_10 import get_run_type
from steps.step_10.step_10_base_scoring import do_base_scoring
from steps.step_10.step_10_filter_methods import do_filter_methods
from steps.step_10.step_10_lasso import do_lasso
from steps.step_10.step_10_plot import create_plots
from steps.step_10.step_10_shap_select import do_shap_select
from steps.step_generic_code.dataframe_colorectal_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging

def run_method_selection(app, run_type, run_all_methods):
    dataframes = get_dataframe(app)
    features = dataframes['X'].columns.tolist()
    if (run_type=="base") or run_all_methods: 
        do_base_scoring(app, dataframes, features, source='isala')
    if (run_type=="filter") or run_all_methods: 
        do_filter_methods(app, dataframes, features, source='isala')
    if (run_type=="lasso") or run_all_methods: 
        do_lasso(app, dataframes, features, source='isala')
    if (run_type=="lasso_steps") or run_all_methods: 
        do_lasso(app, dataframes, features, source='isala', steps=True)
    if (run_type=="shap") or run_all_methods: 
        do_shap_select(app, dataframes, features, source='isala')

def do_step_10(app):
    folder='results/article_8/isala'
    start_logging(folder)
    run_type, run_all_methods = get_run_type()
    if run_type!='plot':
        run_method_selection(app, run_type, run_all_methods)
    create_plots(app, folder, source='isala')