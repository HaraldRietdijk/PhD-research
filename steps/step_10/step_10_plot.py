import matplotlib.pyplot as plt

from steps.step_generic_code.general_functions import check_folder
from steps.step_10.step_10_plot_ranking import plot_ranking_per_model
from steps.step_10.step_10_plot_scoring import plot_score_per_model_per_nr_features_selected, plot_score_per_model_for_optimization

def plot_confusion_matrix():
    # cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                           display_labels=clf.classes_)
    # disp.plot()
    plt.show()

def plot_results(app, folder, nr_of_classes):
    print('Step 10: plotting results')
    folder = folder + '/plots'
    check_folder(folder)
    plot_score_per_model_per_nr_features_selected(app, folder, nr_of_classes)
    plot_ranking_per_model(app, folder, nr_of_classes)

def plot_results_optimization(app, folder):
    print('Step 10: plotting results for optimization')
    folder = folder + '/plots'
    check_folder(folder)
    plot_score_per_model_for_optimization(app, folder)