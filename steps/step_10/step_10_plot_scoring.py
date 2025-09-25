import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import func, and_

from database.models.patient_data import RFE_SCORING, RFE_RESULTS, RFE_SCORING_TYPES

from steps.step_generic_code.general_functions import check_folder

def plot_scoring(scoring_dict, estimator, folder, nr_of_classes, type='Test_Train'):
    def plot_regression_results(scoring, ax1, ax2, left):
        x=scoring['nr_selected']
        if left:
            ax1.set_ylabel('R2 score and \n absolute mean percentage',color='tab:red')
            ax2.set_yticks([])
        else:
            ax1.set_yticks([])
            ax2.set_ylabel('mean square error',color='tab:blue')
        line1 = ax1.plot(x,scoring['r2_score'],color='tab:red', label='r2_score')
        line2 = ax1.plot(x,scoring['mean_absolute_percentage_error'],color='tab:orange', label='mean percentage')
        line3 = ax2.plot(x,scoring['mean_square_error'],color='tab:blue', label='mean square')
        return line1 + line2 + line3

    def plot_classifier_results(scoring, ax1, ax2, left):
        x=scoring['nr_selected']
        if left:
            ax1.set_ylabel('F1 score and \n accuracy',color='tab:red')
            ax2.set_yticks([])
        else:
            ax1.set_yticks([])
        line1 = ax1.plot(x,scoring['f1_score'],color='tab:red', label='f1_score')
        line2 = ax1.plot(x,scoring['accuracy'],color='tab:orange', label='accuracy')
        lines = line1 + line2
        if 'area_under_curve' in scoring.keys():
            if not left:
                ax2.set_ylabel('Area under the curve',color='tab:blue')
            line3 = ax2.plot(x,scoring['area_under_curve'],color='tab:blue', label='area under the curve')
            lines = lines + line3
        return lines

    def create_sub_plot(location, nr_features):
        ax1 = fig.add_subplot(location)
        ax1.set_xticks(np.arange(0, nr_features, step = 4))
        ax1.set_xlabel("Number of features selected")
        ax2 = ax1.twinx()
        return ax1, ax2
    
    def place_labels_and_move_box(ax1, ax2, lines, set_name, left):
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                        box.width, box.height * 0.8])
        ax2.set_position([box.x0, box.y0 + box.height * 0.2,
                        box.width, box.height * 0.8])
        labs = [line.get_label() for line in lines]
        ax1.set_title('Results for '+ set_name + ' set')
        if not left:
            ax1.legend(lines, labs, loc='upper center', bbox_to_anchor=(0.6, -0.15), ncol=2)

    def get_min_max(scoring_dict):
        min_max = {}
        if 'r2_score' in scoring_dict['test'].keys():
            min_max['min_y_l'] = min([0, scoring_dict['test'].loc[:, ['r2_score']].min()[0], 
                           scoring_dict['test'].loc[:, ['mean_absolute_percentage_error']].min()[0], 
                           scoring_dict['train'].loc[:, ['r2_score']].min()[0], 
                           scoring_dict['train'].loc[:, ['mean_absolute_percentage_error']].min()[0]])
            min_max['min_y_r'] = 0.8*min([scoring_dict['test'].loc[:, ['mean_square_error']].min()[0]],
                               scoring_dict['train'].loc[:, ['mean_square_error']].min()[0])
            min_max['max_y_r'] = max(scoring_dict['test'].loc[:, ['mean_square_error']].max()[0],
                          scoring_dict['train'].loc[:, ['mean_square_error']].max()[0])
        else:       
            min_max['min_y_l'] = 0
            min_max['min_y_r'] = 0
            min_max['max_y_r'] = 1.1
        min_max['max_y_l'] = 1.1
        return min_max

    def plot_set(ax1, ax2, min_max, set_name, left):
        scoring = scoring_dict[set_name]
        ax1.set_ylim([min_max['min_y_l'], min_max['max_y_l']])
        ax2.set_ylim([min_max['min_y_r'], min_max['max_y_r']])
        if 'r2_score' in scoring.keys():
            lines1 = plot_regression_results(scoring, ax1, ax2, left)    
        else:
            lines1 = plot_classifier_results(scoring, ax1,ax2, left)
        place_labels_and_move_box(ax1, ax2, lines1, set_name, left)

    def save_scoring_plot():
        title = 'Recursive Feature Elimination for'  + estimator
        if nr_of_classes > 0:
            title += ' for ' +str(nr_of_classes) + ' classes'
        fig.suptitle(title)
        name = folder + '/scoring'
        check_folder(name)
        name = name + '/' + estimator + '_' + str(nr_of_classes) + '.png'
        plt.savefig(name, format='png')

    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(10,5))
    ax1, ax2 = create_sub_plot(121,max(scoring_dict['test']['nr_selected']))
    ax3, ax4 = create_sub_plot(122,max(scoring_dict['test']['nr_selected']))
    min_max = get_min_max(scoring_dict)
    plot_set(ax1, ax2, min_max, 'test', left=True)
    if type=='Test_Train':
        plot_set(ax3, ax4, min_max, 'train', left=False)
    else:
        plot_set(ax3, ax4, min_max, 'optim', left=False)
    save_scoring_plot()

def get_models(app, nr_of_classes):
    models_query = app.session.query(RFE_RESULTS.estimator).filter(RFE_RESULTS.nr_classes==nr_of_classes).distinct()
    models = [row for row in models_query]
    return models

def get_models_from_optimization(app, nr_of_classes):
    models_query = app.session.query(RFE_RESULTS.estimator).join(RFE_SCORING, RFE_RESULTS.id==RFE_SCORING.rfe_results_id)\
                              .filter(and_(RFE_RESULTS.nr_classes==nr_of_classes, RFE_SCORING.test_or_train_data=='optim')).distinct()
    models = [row for row in models_query]
    return models

def build_scoring_queries(app, estimator, nr_of_classes, test_or_train):
    scoring_query = app.session.query(RFE_RESULTS.nr_selected, func.max(RFE_SCORING.value), RFE_SCORING_TYPES.name)\
                               .join(RFE_SCORING, RFE_SCORING.rfe_results_id==RFE_RESULTS.id)\
                               .join(RFE_SCORING_TYPES, RFE_SCORING_TYPES.id==RFE_SCORING.scoring_type_id)\
                               .filter(and_(RFE_RESULTS.estimator==estimator,RFE_RESULTS.nr_classes==nr_of_classes,
                                            RFE_SCORING.test_or_train_data==test_or_train, RFE_RESULTS.nr_splits==1))\
                               .group_by(RFE_RESULTS.nr_selected, RFE_SCORING_TYPES.name, RFE_SCORING.test_or_train_data)\
                               .order_by(RFE_RESULTS.nr_selected)\
                               .all()
    scoring_names_query = app.session.query(RFE_SCORING_TYPES.name)\
                               .join(RFE_SCORING, RFE_SCORING_TYPES.id==RFE_SCORING.scoring_type_id)\
                               .join(RFE_RESULTS, RFE_SCORING.rfe_results_id==RFE_RESULTS.id)\
                               .filter(and_(RFE_RESULTS.estimator==estimator,RFE_RESULTS.nr_classes==nr_of_classes,
                                            RFE_RESULTS.nr_splits==1))\
                               .group_by(RFE_SCORING_TYPES.name)\
                               .all()
    return scoring_query, scoring_names_query

def get_scoring(app, estimator, nr_of_classes, test_or_train):
    scoring_query, scoring_names_query = build_scoring_queries(app, estimator, nr_of_classes,test_or_train)
    first =True
    for name in scoring_names_query:
        if first:
            scoring = pd.DataFrame([(row[0], row[1]) for row in scoring_query if row[2]==name[0]],
                          columns=['nr_selected',name[0]])
            first = False
        else:
            scoring[name[0]] = pd.DataFrame([row[1] for row in scoring_query if row[2]==name[0]])
    return scoring

def plot_score_per_model_per_nr_features_selected(app, folder, nr_of_classes):
    models = get_models(app, nr_of_classes)
    for model in models:
        scoring = {}
        scoring['train']  = get_scoring(app, model[0], nr_of_classes, 'train')
        scoring['test']  = get_scoring(app, model[0], nr_of_classes, 'test')
        plot_scoring(scoring, model[0], folder, nr_of_classes)

def plot_score_per_model_for_optimization(app, folder, nr_of_classes=2):
    models = get_models_from_optimization(app, nr_of_classes)
    for model in models:
        scoring = {}
        scoring['test']  = get_scoring(app, model[0], nr_of_classes, 'test')
        scoring['optim']  = get_scoring(app, model[0], nr_of_classes, 'optim')
        plot_scoring(scoring, model[0], folder, nr_of_classes, type='Test_Optim')