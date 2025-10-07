import matplotlib.pyplot as plt
import numpy as np

from steps.step_generic_code.general_functions import check_folder

def plot_scoring(scoring, model, method, folder):
    plt.rcParams["font.family"] = "serif"
    x = np.arange(1,len(scoring['accuracy'])+1)
    fig, ax1 = plt.subplots()
    ax1.set_xticks(np.arange(1, len(scoring['accuracy'])+1, step = 4))
    ax1.set_xlabel("Nr. of Features used")
    ax1.set_ylabel('F1-Score and Accuracy', fontsize='large', labelpad=30)
    lines = ax1.plot(x, scoring['accuracy'],color='tab:red', label='Accuracy')
    lines = lines + ax1.plot(x, scoring['f1-score'], color='tab:blue', label='F1-score')
    ax1.legend(lines, [line.get_label() for line in lines], loc='lower right')
    title = 'Accuracy and F1-score for ' + model + ' using ' + method
    plt.title(title)
    name = folder + '/' + model + '.png'
    plt.savefig(name, format='png')

def plot_score_per_model_per_nr_features_selected(folder, scores, method):
    for model, model_scores in scores.items():
        scoring = {}
        scoring['accuracy']  = model_scores['accuracy']
        scoring['f1-score']  = model_scores['f1-score']
        plot_scoring(scoring, model, method, folder)

def plot_results_per_method(folder, scores):
    print('Step 10: plotting filter methods results')
    folder = folder + '/plots'
    check_folder(folder)
    for method, scores_per_method in scores.items():
        method_folder = folder + '/' + method
        check_folder(method_folder)
        plot_score_per_model_per_nr_features_selected(method_folder, scores_per_method, method)

