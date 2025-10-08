from sqlalchemy.sql import func, or_

import matplotlib.pyplot as plt
import numpy as np

from database.models.feature_selection_data import SELECTION_METHOD, METHOD_RESULTS, METHOD_RESULTS_FEATURES
from steps.step_generic_code.general_functions import check_folder

def plot_scoring(scores, method, folder, model = None, thresholds=None):
    plt.rcParams["font.family"] = "serif"
    fig, ax1 = plt.subplots()
    if thresholds:
        x = thresholds
        ax1.set_xticks(thresholds)
        ax1.set_xlabel("Threshold used")
    else:
        x = np.arange(1,len(list(scores.values())[0][0])+1)
        ax1.set_xticks(np.arange(1, len(x), step = 4))
        ax1.set_xlabel("Nr. of Features used")
    ax1.set_ylabel('Mertic score', fontsize='large', labelpad=30)
    lines = None
    for label, metric in scores.items():
        if not lines:
            lines = ax1.plot(x, metric[0], color=metric[1], label=label)
        else:
            lines = lines + ax1.plot(x, metric[0], color=metric[1], label=label)
    ax1.legend(lines, [line.get_label() for line in lines], loc='lower right')
    title = 'Metrics using ' + method
    name = folder + '/' + method + '.png'
    if model:
        title = title + ' for ' + model
        name = folder + '/' + model + '.png'
    plt.title(title)
    plt.savefig(name, format='png')

def plot_score_per_model_per_nr_features_selected(folder, scores, method, thresholds=None):
    for model, model_scores in scores.items():
        scoring = {}
        scoring['accuracy']  = (model_scores['accuracy'], 'tab:red')
        scoring['f1-score']  = (model_scores['f1-score'], 'tab:blue')
        plot_scoring(scoring, method, folder, model, thresholds=thresholds)

def get_scores_per_method(app, run_id):
    scores_per_method = {}
    average_accuracy = app.session.query(SELECTION_METHOD.name, METHOD_RESULTS.nr_features, METHOD_RESULTS.threshold,
                                         func.avg(METHOD_RESULTS.accuracy), func.avg(METHOD_RESULTS.f1_score),
                                         func.avg(METHOD_RESULTS.precision), func.avg(METHOD_RESULTS.recall))\
                                  .join(METHOD_RESULTS, SELECTION_METHOD.id==METHOD_RESULTS.method_id)\
                                  .filter(or_(METHOD_RESULTS.run_id==run_id, run_id==-1))\
                                  .group_by(SELECTION_METHOD.name, METHOD_RESULTS.threshold, METHOD_RESULTS.nr_features)\
                                  .order_by(SELECTION_METHOD.name, METHOD_RESULTS.threshold, METHOD_RESULTS.nr_features).all()
    for row in average_accuracy:
        if not row[0] in scores_per_method.keys():
            scores_per_method[row[0]] = {'accuracy' : ([],'tab:red'), 'f1-score' : ([],'tab:blue'), 
                                         'precision' : ([],'tab:orange'), 'recall' : ([],'tab:green')}
        scores_per_method[row[0]]['accuracy'][0].append(row[3])
        scores_per_method[row[0]]['f1-score'][0].append(row[4])
        scores_per_method[row[0]]['precision'][0].append(row[5])
        scores_per_method[row[0]]['recall'][0].append(row[6])
    return scores_per_method

def plot_results_per_method(folder, scores, run_id, thresholds = None):
    print('Step 10: plotting filter methods results')
    folder = folder + '/plots'
    check_folder(folder)
    for method, scores_per_method in scores.items():
        method_folder = folder + '/' + method + "_" + str(run_id)
        check_folder(method_folder)
        plot_score_per_model_per_nr_features_selected(method_folder, scores_per_method, method, thresholds=thresholds)

def plot_average_per_method(app, folder, run_id=-1, thresholds = None):
    folder = folder +'/plots/average/run_' + str(run_id)
    check_folder(folder)
    scores_per_method = get_scores_per_method(app, run_id)
    for method, method_scores in scores_per_method.items():
        plot_scoring(method_scores, method, folder, thresholds=thresholds)
