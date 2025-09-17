import matplotlib.pyplot as plt

from steps.step_generic_code.general_functions import check_folder

def plot_boxplot_for_feature(feature, results, folder):
    fig, ax = plt.subplots()
    fig.suptitle('Test-Train feature comparison for ' + feature)
    ax.bxp(results, showfliers=True)
    ax.set_ylabel("value")
    name = folder + '/boxplots'
    check_folder(name)
    name = name + '/' + feature + '.png'
    plt.savefig(name, format='png')

def get_result_values(quantiles, mean, outliers, label):
    result_values = {
            'label' : label,
            'whislo': quantiles.iloc[0],    # Bottom whisker position
            'q1'    : quantiles.iloc[1],    # First quartile (25th percentile)
            'med'   : mean,                 # Median         (50th percentile)
            'q3'    : quantiles.iloc[2],    # Third quartile (75th percentile)
            'whishi': quantiles.iloc[3],    # Top whisker position
            'fliers': outliers              # Outliers
        }
    return result_values

def get_measures(dataframe):
    result_quantiles = dataframe.quantile([0.05,.25,.75, 0.95])
    results_mean = dataframe.mean()
    results_variance = dataframe.var()
    return (result_quantiles, results_mean, results_variance)

def get_outliers(dataframe, result_quantiles, feature):
    results_outlier_bottom = dataframe.loc[dataframe[feature] < result_quantiles[feature].iloc[0]][feature].to_list() 
    results_outlier_top = dataframe.loc[dataframe[feature] > result_quantiles[feature].iloc[3]][feature].to_list()
    return results_outlier_bottom + results_outlier_top 

def plot_boxplots(dataframes, features, folder):
    results = {}
    results['X_test'] = get_measures(dataframes['X_test'])
    results['X_train'] = get_measures(dataframes['X_train'])
    for feature in features:
        box_results = []
        outliers = get_outliers(dataframes['X_test'], results['X_test'][0], feature)
        box_results.append(get_result_values(results['X_test'][0][feature],results['X_test'][1][feature], outliers, 'Test'))
        outliers = get_outliers(dataframes['X_train'], results['X_train'][0], feature)
        box_results.append(get_result_values(results['X_train'][0][feature],results['X_train'][1][feature], outliers, 'Train'))
        plot_boxplot_for_feature(feature, box_results, folder)

