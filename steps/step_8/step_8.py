from database.models.patient_data import RTW_CLASSES

from steps.step_generic_code.general_functions import start_logging
from steps.step_generic_code.dataframe_knee_operations import get_dataframe

from steps.step_8.step_8_selection_regressor import regressor_selection, nn_regressor_selection
from steps.step_8.step_8_selection_classifier import classifier_selection
from steps.step_8.step_8_RFE_regressor import RFE_regressors
from steps.step_8.step_8_RFE_classifier import RFE_classifiers, RFE_classifiers_optimize, RFE_classifiers_optimize_top
from steps.step_8.step_8_plot import plot_results, plot_results_optimization
from steps.step_8.step_8_ranking_scoring import get_ranking_score_per_model_for_set

def get_nr_classes_options(app):
    # nr_classes_query = app.session.query(RTW_CLASSES.nr_of_classes).distinct()
    # return [row[0] for row in nr_classes_query]
    return [0,2]

def do_step_8(app):
    base_folder='results/fs_one_step'
    start_logging(base_folder)
    for nr_of_classes in get_nr_classes_options(app):
        dataframes = get_dataframe(app, nr_of_classes)
        if nr_of_classes==0:
            folder = base_folder + '/regressors'
            fitted_models = regressor_selection(app, folder, dataframes)
            nn_regressor_selection(app, base_folder, dataframes)
            RFE_regressors(app, dataframes, nr_of_classes, fitted_models)
        else:
            folder = base_folder + '/classifiers'
            fitted_models = classifier_selection(app, folder, dataframes, nr_of_classes)
            RFE_classifiers(app, dataframes, nr_of_classes, fitted_models)
            RFE_classifiers_optimize_top(app, dataframes, fitted_models)
            plot_results(app, folder, nr_of_classes)
            get_ranking_score_per_model_for_set(app,folder,'test')
    folder = base_folder + '/optimize_top'
    dataframes = get_dataframe(app, 2)    
    RFE_classifiers_optimize(app, dataframes)
    plot_results_optimization(app, folder)
    get_ranking_score_per_model_for_set(app,folder,'optim')
