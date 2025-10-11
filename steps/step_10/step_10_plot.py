from sqlalchemy.sql import func, or_, and_
from sqlalchemy.orm import aliased

import matplotlib.pyplot as plt
import numpy as np

from database.models.feature_selection_data import SELECTION_METHOD, METHOD_RESULTS
from steps.step_generic_code.general_functions import check_folder

COLORS = ['tab:red','tab:blue','tab:orange','tab:green']

def plot_scoring(scores, folder, title, name):
    def determine_x():
        x = []
        only_thresholds = False
        features = False
        for data_for_line in scores.values():
            if len(data_for_line['x'])>len(x):
                x = data_for_line['x']
            if len(x)<43:
                threshold_values = data_for_line['threshold']
                only_thresholds = not features
            else:
                features = True
                only_thresholds = False
            if only_thresholds:
                x = [value/min(threshold_values) for value in threshold_values]
        return x, only_thresholds
    
    def set_x_labels():
        if thresholds:
            ax1.set_xticks(x, labels=line_data['threshold'])
            ax1.set_xlabel("Threshold used")
        else:
            ax1.set_xticks(np.arange(1, len(x), step = 4))
            ax1.set_xlabel("Nr. of Features used")

    plt.rcParams["font.family"] = "serif"
    fig, ax1 = plt.subplots()
    x, thresholds = determine_x()
    ax1.set_ylabel('Mertic score', fontsize='large', labelpad=30)
    lines = None
    for label, line_data in scores.items():
        x_used = x
        if (len(line_data['y'])<len(x)) and not thresholds:
            x_used = line_data['x']
        if not lines:
            set_x_labels()
            lines = ax1.plot(x_used, line_data['y'], color=line_data['color'], label=label)
        else:
            lines = lines + ax1.plot(x_used, line_data['y'], color=line_data['color'], label=label)
    ax1.legend(lines, [line.get_label() for line in lines], loc='lower right')
    name = folder + '/' + name + '.png'
    plt.title(title)
    plt.savefig(name, format='png')

def proces_query_results(scores_per_method, average_accuracy):
    x_method = {}
    for row in average_accuracy:
        if not row[0] in scores_per_method.keys():
            scores_per_method[row[0]] = {'accuracy' : {'y':[],'color':COLORS[0]}, 'f1-score' : {'y':[],'color':COLORS[1]}, 
                                         'precision' : {'y':[],'color':COLORS[2]}, 'recall' : {'y':[],'color':COLORS[3]}}
            x_method[row[0]] = {'x' : [], 'threshold' : []}
        scores_per_method[row[0]]['accuracy']['y'].append(row[3])
        scores_per_method[row[0]]['f1-score']['y'].append(row[4])
        scores_per_method[row[0]]['precision']['y'].append(row[5])
        scores_per_method[row[0]]['recall']['y'].append(row[6])
        x_method[row[0]]['x'].append(row[1])
        x_method[row[0]]['threshold'].append(row[2])
    for method in x_method.keys():
        for metric_scores in scores_per_method[method].values():
            metric_scores['x'] = x_method[method]['x']
            metric_scores['threshold'] = x_method[method]['threshold']
    return scores_per_method

def get_threshold_methods(app):
    RESULTS_2 = aliased(METHOD_RESULTS)
    print('Getting threshold methods')
    threshold_methods_query = app.session.query(METHOD_RESULTS.method_id).distinct(METHOD_RESULTS.method_id)\
                                   .join(RESULTS_2,and_(METHOD_RESULTS.method_id==RESULTS_2.method_id,
                                                        METHOD_RESULTS.threshold==RESULTS_2.threshold))\
                                   .filter(and_(METHOD_RESULTS.nr_features<RESULTS_2.nr_features,
                                                METHOD_RESULTS.threshold>0)).all()   
    return [row[0] for row in threshold_methods_query]

def get_scores_per_method(app, threshold_methods, run_id, model='all'):
    scores_per_method = {'OnFeature' : {}, 'OnThreshold' : {}}
    print('Collecting scores for ', model)
    average_accuracy = app.session.query(SELECTION_METHOD.name, METHOD_RESULTS.nr_features, METHOD_RESULTS.threshold,
                                         func.avg(METHOD_RESULTS.accuracy), func.avg(METHOD_RESULTS.f1_score),
                                         func.avg(METHOD_RESULTS.precision), func.avg(METHOD_RESULTS.recall))\
                                  .join(METHOD_RESULTS, SELECTION_METHOD.id==METHOD_RESULTS.method_id)\
                                  .filter(and_(or_(METHOD_RESULTS.run_id==run_id, run_id==-1),
                                               or_(METHOD_RESULTS.model==model, model=='all'),
                                               METHOD_RESULTS.method_id.notin_(threshold_methods)))\
                                  .group_by(SELECTION_METHOD.name, METHOD_RESULTS.nr_features, METHOD_RESULTS.threshold)\
                                  .order_by(SELECTION_METHOD.name, METHOD_RESULTS.nr_features).all()
    scores_per_method['OnFeature'] = proces_query_results(scores_per_method['OnFeature'], average_accuracy)
    # scores_per_method = proces_query_results(scores_per_method, average_accuracy)
    threshold_accuracy = app.session.query(SELECTION_METHOD.name, func.avg(METHOD_RESULTS.nr_features), METHOD_RESULTS.threshold,
                                         func.avg(METHOD_RESULTS.accuracy), func.avg(METHOD_RESULTS.f1_score),
                                         func.avg(METHOD_RESULTS.precision), func.avg(METHOD_RESULTS.recall))\
                                  .join(METHOD_RESULTS, SELECTION_METHOD.id==METHOD_RESULTS.method_id)\
                                  .filter(and_(or_(METHOD_RESULTS.run_id==run_id, run_id==-1),
                                               or_(METHOD_RESULTS.model==model, model=='all'),
                                               METHOD_RESULTS.method_id.in_(threshold_methods)))\
                                  .group_by(SELECTION_METHOD.name, METHOD_RESULTS.threshold)\
                                  .order_by(SELECTION_METHOD.name, METHOD_RESULTS.threshold).all()
    scores_per_method['OnThreshold'] = proces_query_results(scores_per_method['OnThreshold'], threshold_accuracy)
    return scores_per_method

def get_scores_per_model(app, threshold_methods, run_id):
    scores_per_model = {}
    models = app.session.query(METHOD_RESULTS.model).distinct(METHOD_RESULTS.model).all()
    for model in models:
        scores_per_model[model[0]] = get_scores_per_method(app, threshold_methods, run_id, model=model[0])
    return scores_per_model

def plot_metrics_over_all_models_per_method(scores_per_method, folder, run_id=-1):
    folder = folder +'/plots/metric_per_method'
    if run_id>-1:
        folder = folder +'/run_' + str(run_id)
    check_folder(folder)
    for scores_per_type in scores_per_method.values():
        for method, method_scores in scores_per_type.items():
            title = 'Average metrics for ' + method
            name = 'metrics_for_' + method
            plot_scoring(method_scores, folder, title, name)

def get_accuracy_for_all_methods(scores_per_method):
    accuracy_all_methods = {'OnFeature' : {}, 'OnThreshold' : {}}
    for method_type, scores_per_type in scores_per_method.items():
        for idx, method_scores in enumerate(scores_per_type.items()):
            accuracy_all_methods[method_type][method_scores[0]] = method_scores[1]['accuracy']
            accuracy_all_methods[method_type][method_scores[0]]['color'] = COLORS[idx]
    return accuracy_all_methods

def plot_accuracy_for_all_methods_per_features(scores_per_method, folder):
    # one graph with four lines, one line per method, average over all models
    folder = folder +'/plots/metric_per_method'
    check_folder(folder)
    accuracy_all_methods = get_accuracy_for_all_methods(scores_per_method)
    plot_scoring(accuracy_all_methods['OnFeature'], folder, title='Accuracy for all methods', name='accuracy_all_methods')

def plot_metrics_per_model_per_method(scores_per_model, folder, run_id=-1):
    # 32 graphs (4 methods * 8 models) with four lines, one per metric
    folder = folder +'/plots/metric_per_model'
    if run_id>-1:
        folder = folder +'/run_' + str(run_id)
    check_folder(folder)
    for model, scores_per_method in scores_per_model.items():
        for method, method_scores in scores_per_method['OnThreshold'].items():
            title = 'Average metrics for ' + model + ' using ' + method
            name = 'metrics_for_' + model + '_' + method
            model_folder = folder + '/' + model
            check_folder(model_folder)
            plot_scoring(method_scores, model_folder, title, name)

def plot_accuracy_for_all_methods_per_model(scores_per_model, folder, run_id=-1):
    # 8 graphs (8 models) with four lines, one per method
    folder = folder +'/plots/metric_per_model'
    if run_id>-1:
        folder = folder +'/run_' + str(run_id)
    check_folder(folder)
    for model, scores_per_method in scores_per_model.items():
        accuracy_all_methods = get_accuracy_for_all_methods(scores_per_method)
        title = 'Average metrics for ' + model + ' all methods'
        name = 'metrics_for_' + model + '_all'
        model_folder = folder + '/' + model
        check_folder(model_folder)
        plot_scoring(accuracy_all_methods['OnFeature'], model_folder, title, name)

def create_plots(app, folder, run_id=-1):
    threshold_methods = get_threshold_methods(app)
    scores_per_method = get_scores_per_method(app, threshold_methods, run_id)
    scores_per_model = get_scores_per_model(app, threshold_methods, run_id)
    plot_metrics_over_all_models_per_method(scores_per_method, folder) # 4 graphs (4 methods) with each four lines, one line per metric, taking the average metric over all models
    plot_accuracy_for_all_methods_per_features(scores_per_method, folder) # one graph with four lines, one line per method, average over all models
    plot_metrics_per_model_per_method(scores_per_model, folder) # 24 graphs (3*8 models) with four lines, one per metric
    plot_accuracy_for_all_methods_per_model(scores_per_model, folder, run_id) # 24 graphs (3*8 models) with four lines, one per metric

