import os
import sys

from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging

from steps.step_10.step_10_selection_classifier import classifier_selection

from steps.step_generic_code.general_variables.general_variables_all_shap import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS
from steps.step_10.step_10_generic_selection import save_pickle_and_metrics

from steps.step_10.step_10_plot import plot_results

import shap

import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from shap_select import shap_select
except ModuleNotFoundError:
    root = os.path.realpath("..")
    sys.path.append(root)
    from shap_select import shap_select


# === Recognize models that requires masker/callable ===
def is_tree_model(model) -> bool:
    # OK out-of-the-box: DT, RF, GBC
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)):
        return True
    # ADA: ok if base learner is a tree (default: DecisionTree stump)
    if isinstance(model, AdaBoostClassifier):
        # scikit-learn >=1.2 usa .estimator, legacy .base_estimator
        base = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
        return (base is None) or isinstance(base, (DecisionTreeClassifier, DecisionTreeRegressor))
    return False


def do_step_10(app):
    # Settings
    num_of_classes = 2
    base_folder = 'results/fs_shap'

    start_logging(base_folder)

    dataframes = get_dataframe(app, num_of_classes)
    features = dataframes['X'].columns.tolist()

    # Train + plot
    folder_clf = base_folder + '/classifiers'
    fitted_models = classifier_selection(app, folder_clf, dataframes, num_of_classes)
    plot_results(app, folder_clf, num_of_classes)

    # --- 1) Binarization (under/above mean) ---
    y_cont = dataframes['Y']
    thr = y_cont.mean()
    y_bin = (y_cont >= thr).astype(int)
    print("Mean", float(thr))
    # dont overwrite original Y
    Y_for_shap = y_bin

    task = "binary"

    # Results folder (Work in Progress)
    folder = base_folder + '/shap_select'
    os.makedirs(folder, exist_ok=True)

    print("Fitted models:", fitted_models)

    for model_name, model in fitted_models.items():
        model_type = type(model).__name__

        # Skip models that breaks shap ---
        if not is_tree_model(model):
            print(f"[SKIP] {model_name} ({model_type}) no-tree-based → require masker/callable. Skip.")
            continue
        # b) AdaBoost avoidance because cause TypeError
        if isinstance(model, AdaBoostClassifier):
            print(f"[SKIP] {model_name} (AdaBoost) → TypeError. Skip.")
            continue

        print(f"[SHAP-SELECT] Execute su {model_name} ({model_type})")

        results = shap_select(
            tree_model=model,
            validation_df=dataframes['X'],
            target=Y_for_shap,
            feature_names=features,
            task=task,
            threshold=0.05,
            return_extended_data=True
        )

        significance_df, shap_values_df = results

        # --- print chosen and discarded features ---
        selected = significance_df.loc[significance_df["selected"] == 1, "feature name"].tolist()
        dropped  = significance_df.loc[significance_df["selected"] == 0, "feature name"].tolist()
        neg_coef = significance_df.loc[significance_df["selected"] == -1, "feature name"].tolist()

        print(f"\nResults for {model_name}:")
        print(" Chosen:", selected)
        print(" Discarded:", dropped)
        if neg_coef:
            print(" Negative Coefficients:", neg_coef)
        print("-" * 60)
