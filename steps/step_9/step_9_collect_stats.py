import pandas as pd
from steps.step_generic_code.general_functions import check_folder
from steps.step_9.step_9_OLS import kruskal_wallis_test

def get_dataset_measures(dataframe, features, folder):
    means = pd.DataFrame([dataframe['X'].mean()])
    mins = pd.DataFrame([dataframe['X'].min()])
    maxs = pd.DataFrame([dataframe['X'].max()])
    stds = pd.DataFrame([dataframe['X'].std(ddof=0)])
    kw = kruskal_wallis_test(dataframe)
    results = {}
    for feature in features:
        results[feature] = [means[feature][0], mins[feature][0], maxs[feature][0], stds[feature][0], kw[feature][1]] 
    folder = folder + '/stats'
    check_folder(folder)
    file = folder + '/feature_stats.csv'
    pd.DataFrame.from_dict(results, orient='index', columns=['mean','min','max','stdev','kw_p_value']).to_csv(file)
