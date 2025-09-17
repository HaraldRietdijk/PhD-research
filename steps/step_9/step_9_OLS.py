import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import kstest, kruskal, norm, mannwhitneyu

from steps.step_generic_code.general_functions import check_folder

def ols_model_info(X, X_test, Y, Y_test):
    model = sm.OLS(Y.astype(float), X.astype(float)).fit()
    print(model.summary())
    LRresult = (model.summary().tables[1])
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    print(r2)

def glm_model_info(X, X_test, Y, Y_test):
    model = sm.GLM(Y.astype(float), X.astype(float)).fit()
    print(model.summary())
    LRresult2 = (model.summary().tables[1])
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    print(r2)

def get_X_Y_frames(dataframes):
    X = sm.add_constant(dataframes['X_train'])
    X_test = sm.add_constant(dataframes['X_test'])
    Y = dataframes['Y_train']
    Y_test = dataframes['Y_test']
    return X, X_test, Y, Y_test

def print_correlation_matrix(dataframes, folder):
    correlation_matrix = dataframes['X'].corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    name = folder + '/correlationplots'
    check_folder(name)
    name = name + '/corrrelation_all_features.png'
    plt.savefig(name, format='png')

def kruskal_wallis_test(dataframes):
    def kruskal_wallis_test(column):
        test = dataframes['X_test'][column].dropna()
        train = dataframes['X_train'][column].dropna()
        stat, p_value = kruskal(test, train)
        return stat, p_value

    kruskal_results = {column: kruskal_wallis_test(column) for column in dataframes['X'].columns}
    for column, (stat, p_value) in kruskal_results.items():
        print(f"Kruskal-Wallis Test for '{column}'")
        print(f"Statistic: {stat}, p-value: {p_value}")
        if p_value <= 0.05:
            print("There is a significant difference between Train and Test patients for the '{}' metric.".format(column))
            print()
    return kruskal_results   

def mann_whitney_u_test(dataframes):
    def mannwhitneyu_test(column):
        test = dataframes['X_test'][column].dropna()
        train = dataframes['X_train'][column].dropna()
        stat, p_value = mannwhitneyu(test.astype(float), train.astype(float))
        return stat, p_value

    mannwhitneyu_results = {column: mannwhitneyu_test(column) for column in dataframes['X'].columns}
    for column, (stat, p_value) in mannwhitneyu_results.items():
        print(f"Mann-whitney-u Test for '{column}'")
        print(f"Statistic: {stat}, p-value: {p_value}")
        if p_value <= 0.05:
            print("There is a significant difference between Train and Test patients for the '{}' metric.".format(column))
            print()    

def datasets_info(dataframes, folder):
    X, X_test, Y, Y_test = get_X_Y_frames(dataframes)
    # ols_model_info(X, X_test, Y, Y_test)
    # glm_model_info(X, X_test, Y, Y_test)
    # print_correlation_matrix(dataframes, folder)
    kruskal_wallis_test(dataframes)
    mann_whitney_u_test(dataframes)
