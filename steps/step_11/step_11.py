from numba import jit, cuda

from steps.step_11.step_11_optimizations import do_optim_methods_runs
from steps.step_generic_code.general_functions import start_logging
from steps.step_11.step_11_model_average import fill_model_averages
from steps.step_11.step_11_model_scoring import fill_model_scoring, fill_s_ordering
from steps.step_11.step_11_feature_scoring import fill_feature_scoring
from steps.step_11.step_11_orderings import accuracy_delta_based_ordering, create_or_update_rankings_info,\
                                            s_ordering_contribution_weighted, simple_average, weighted_average

def fill_metrics(app, optim):
    print("Filling metrics")
    fill_model_averages(app, optim)
    fill_model_scoring(app, optim)
    fill_feature_scoring(app, optim)
    print("Step 8: Filling S_ordering")
    fill_s_ordering(app, optim)
    print("Step 9: Rankings Info")
    create_or_update_rankings_info(app, optim)

def determine_optimisation_orderings(app):
    print("Simple Average")
    simple_average(app)
    print("Weighted Average")
    weighted_average(app)
    print("Accuracy_Delta  based ")
    accuracy_delta_based_ordering(app)
    print("s_Ordering contribution  based ")
    s_ordering_contribution_weighted(app)

def do_step_11(app):
    folder='results/fs_optim'
    start_logging(folder)
    # Fill metrics of standard orderings
    # fill_metrics(app, optim=False)
    determine_optimisation_orderings(app)
    do_optim_methods_runs(app)
    # Fill metrics of optimization orderings
    fill_metrics(app, optim=True)