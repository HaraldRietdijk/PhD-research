import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import and_, or_, not_, func

from database.models.hft_tables import HFT_ALGORITHM_T, HFT_MODEL_T, HFT_MODEL_PARAMETERS_T,\
                                       HFT_METRICS_T, HFT_METRICS_GENERAL_MODELT
from steps.step_generic_code.general_functions import start_logging

def plot_graph(results, names, title, labels):
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()+0.01
            ax.text(rect.get_x()+rect.get_width()/2., h, '%2.2f'%h    ,
                    ha='center', va='bottom', size='large')

    plt.rcParams["font.family"] = "serif"
    x = np.arange(len(results[0]))
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize='large')
    axis = plt.gca()
    axis.set_ylim([0, 1])
    plt.grid(axis = 'y')
    plt.ylabel('F1-Score and Accuracy', fontsize='large', labelpad=30) 
    move = 0.18
    colors = ['tab:blue','tab:cyan','tab:orange','tab:red']
    rects=[]
    for i in range(4):
        rects.append(ax.bar(x-0.27+i*move, results[i], width=.12, color=colors[i], align='center', label=labels[i]))
        autolabel(rects[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, size='x-large', ha='right', rotation_mode='anchor')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.3,
                    box.width, box.height * 0.7])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize='large')
    plt.show()

def transform_datasets(data_sets):
    results=[[],[],[],[]]
    names=[]
    for model in data_sets[0].keys():
        names.append(model)
        results[0].append(data_sets[0][model]['accuracy'])
        results[1].append(data_sets[1][model]['accuracy'])
        results[2].append(data_sets[0][model]['f1'])
        results[3].append(data_sets[1][model]['f1'])
    return names, results

def get_data_sets_compare_acc_F1(app, run_id, step):
    data_set = {}
    if step == 2:
        scores = app.session.query(HFT_METRICS_GENERAL_MODELT.score, HFT_METRICS_GENERAL_MODELT.scoring, 
                                   HFT_METRICS_GENERAL_MODELT.algorithm)\
                            .filter(HFT_METRICS_GENERAL_MODELT.hft_run_id==run_id)
        for score, scoring, algorithm in scores:
            if not (algorithm in data_set.keys()):
                data_set[algorithm] = {}
            data_set[algorithm][scoring] = score
    else:
       scores = app.session.query(func.avg(HFT_METRICS_T.accuracy),func.avg(HFT_METRICS_T.f1_score), HFT_MODEL_T.algorithm)\
                           .join(HFT_MODEL_T).filter(HFT_METRICS_T.hft_run_id==run_id).group_by(HFT_MODEL_T.algorithm).all()
       for accuracy, f1, algorithm in scores:
           data_set[algorithm] = {'accuracy': accuracy, 'f1': f1}       
    return data_set

def do_step_3_graph(app):
    folder = 'results/article_1/plots'
    start_logging(folder)
    data_sets = []
    data_sets.append(get_data_sets_compare_acc_F1(app, 16, 3))
    data_sets.append(get_data_sets_compare_acc_F1(app, 14, 3))
    title = 'Comparison of VFC and NS results'
    labels = ['VFC accuracy', 'NS accuracy', 'VFC F1-score', 'NS F1-score']
    names, results = transform_datasets(data_sets)
    plot_graph(results, names, title, labels)
