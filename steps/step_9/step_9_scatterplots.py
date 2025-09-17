import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from steps.step_generic_code.general_functions import check_folder

def plot_scatterplot_for_feature(feature, X_test, X_train, Y_test, Y_train, min_max, folder):
    fig = plt.figure()
    fig.suptitle('Test-Train feature comparison for ' + feature)
    ax1 = fig.add_subplot(121)
    ax1.scatter(X_test, Y_test, color ='orange')
    ax1.set_ylabel("Return to Work weeks")
    ax1.set_xlabel("Test")
    ax1.set_xlim([min_max['xmin'], min_max['xmax']])
    ax1.set_ylim([min_max['ymin'], min_max['ymax']])
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2 = fig.add_subplot(122)
    ax2.scatter(X_train, Y_train, color ='blue')
    ax2.set_xlabel("Train")
    ax2.set_xlim([min_max['xmin'], min_max['xmax']])
    ax2.set_ylim([min_max['ymin'], min_max['ymax']])
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    name = folder + '/scatterplots'
    check_folder(name)
    name = name + '/' + feature + '.png'
    plt.savefig(name, format='png')

def add_scatterplot_for_feature(fig, feature, X_train, Y_train, min_max, index):
    ax = fig.add_subplot(7,7,index)
    ax.set_xlim([min_max['xmin_c'], min_max['xmax_c']])
    ax.set_ylim([min_max['ymin_c'], min_max['ymax_c']])
    ax.scatter(X_train, Y_train, color ='blue')
    ax.set_xlabel(feature)
    return fig

def get_figure_for_matrix():
    fig = plt.figure(figsize=(100,100), num=1000)
    fig.suptitle('Test-Train feature comparison')
    return fig

def plot_scatterplots(dataframes, features, folder):
    Y_test = dataframes['Y_test'].to_list()
    Y_train = dataframes['Y_train'].to_list()
    min_max = {}
    min_max['ymin'] = 0.9 * min([min(Y_test),min(Y_train)])
    min_max['ymax'] = 1.1 * max([max(Y_test),max(Y_train)])
    min_max['ymin_c'] = 0.9 * min(Y_train)
    min_max['ymax_c'] = 1.1 * max(Y_train)
    fig = get_figure_for_matrix()
    index = 1
    for feature in features:
        X_test = dataframes['X_test'][feature].to_list()
        X_train = dataframes['X_train'][feature].to_list()
        min_max['xmin'] = 0.9 * min([min(X_test),min(X_train)])
        min_max['xmax'] = 1.1 * max([max(X_test),max(X_train)])
        min_max['xmin_c'] = 0.9 * min(X_train)
        min_max['xmax_c'] = 1.1 * max(X_train)
        plot_scatterplot_for_feature(feature, X_test, X_train, Y_test, Y_train, min_max, folder)
        fig = plt.figure(num=1000)
        fig = add_scatterplot_for_feature(fig, feature, X_train, Y_train, min_max, index)
        index += 1
    name = folder + '/scatterplots'
    check_folder(name)
    name = name + '/All_Train.png'
    fig = plt.figure(num=1000)
    plt.savefig(name, format='png')
    fig.show()