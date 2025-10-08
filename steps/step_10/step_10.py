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

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

try:
    from shap_select import shap_select
except ModuleNotFoundError:
    root = os.path.realpath("..")
    sys.path.append(root)
    from shap_select import shap_select


# Avoid models that are not tree-based
# Remove in future versions once solved compatibility problema
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


# continuous score for AUC (if available)
def _decision_scores_or_proba(model, X):
    """Return a continuous score: predict_proba[:,1] if available,
    otherwise decision_function, otherwise predictions (fallback)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2:
            # binary or multi-class
            if proba.shape[1] == 2:
                return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    # fallback
    return model.predict(X)


# Compute & print metrics
def compute_metrics(model, X, y, average: str = "binary") -> dict:
    """Compute standard metrics for classification."""
    y_pred = model.predict(X)
    scores = _decision_scores_or_proba(model, X)

    # classic metrics
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average=average, zero_division=0)
    rec  = recall_score(y, y_pred, average=average, zero_division=0)
    f1   = f1_score(y, y_pred, average=average, zero_division=0)

    # AUC (if possible)
    try:
        auc = roc_auc_score(y, scores)
    except Exception:
        auc = None

    cm = confusion_matrix(y, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "y_pred": y_pred,
        "scores": scores,
    }


def _fmt_auc(x):
    return f"{x:.4f}" if (x is not None) else "n/a"


def print_metrics_block(model_name: str, mt_train: dict, mt_test: dict) -> None:
    """Print a block of metrics for train and test."""
    print(f"\n[METRICS] {model_name}")
    print("  Train:",
          f"Acc={mt_train['accuracy']:.4f}",
          f"Prec={mt_train['precision']:.4f}",
          f"Rec={mt_train['recall']:.4f}",
          f"F1={mt_train['f1']:.4f}",
          f"AUC={_fmt_auc(mt_train['auc'])}")
    print("   CM Train:\n", mt_train["cm"])

    print("  Test :",
          f"Acc={mt_test['accuracy']:.4f}",
          f"Prec={mt_test['precision']:.4f}",
          f"Rec={mt_test['recall']:.4f}",
          f"F1={mt_test['f1']:.4f}",
          f"AUC={_fmt_auc(mt_test['auc'])}")
    print("   CM Test:\n", mt_test["cm"])


def evaluate_and_report(model_name, model, X_train, y_train, X_test, y_test, average: str = "binary"):
    """Wrapper to compute and print metrics for train and test sets."""
    mt_train = compute_metrics(model, X_train, y_train, average=average)
    mt_test  = compute_metrics(model, X_test,  y_test,  average=average)
    print_metrics_block(model_name, mt_train, mt_test)
    return mt_train, mt_test


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

    # Binarize target for SHAP (mean threshold)
    y_cont = dataframes['Y']
    thr = y_cont.mean()
    y_bin = (y_cont >= thr).astype(int)
    print("Mean", float(thr))
    Y_for_shap = y_bin

    task = "binary"

    # Results folder (Work in Progress) 
    folder = base_folder + '/shap_select'
    os.makedirs(folder, exist_ok=True)

    print("Fitted models:", fitted_models)

    for model_name, model in fitted_models.items():
        # Pre-score before selection
        evaluate_and_report(
            model_name=model_name,
            model=model,
            X_train=dataframes['X_train'],
            y_train=Y_for_shap.loc[dataframes['X_train'].index],  # allinea per indice
            X_test=dataframes['X_test'],
            y_test=Y_for_shap.loc[dataframes['X_test'].index],    # allinea per indice
            average="binary"
        )

        model_type = type(model).__name__

        # Skip non-tree-based models
        if not is_tree_model(model):
            print(f"[SKIP] {model_name} ({model_type}) no-tree-based → require masker/callable. Skip.")
            continue
        if isinstance(model, AdaBoostClassifier):
            print(f"[SKIP] {model_name} (AdaBoost) → TypeError. Skip.")
            continue
        if isinstance(model, RandomForestClassifier):
            print(f"[SKIP] {model_name} (RandomForest) → TypeError. Skip.")
            continue
        if isinstance(model, DecisionTreeClassifier):
            print(f"[SKIP] {model_name} (DecisionTree) → TypeError. Skip.")
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

        # print choosen/discarded features
        selected = significance_df.loc[significance_df["selected"] == 1, "feature name"].tolist()
        dropped  = significance_df.loc[significance_df["selected"] == 0, "feature name"].tolist()
        neg_coef = significance_df.loc[significance_df["selected"] == -1, "feature name"].tolist()

        print(f"\nResults for {model_name}:")
        print(" Chosen:", selected)
        print(" Discarded:", dropped)
        if neg_coef:
            print(" Negative Coefficients:", neg_coef)
        print("-" * 60)

        # feature selection (fallback if none pass the threshold)
        selected = significance_df.loc[significance_df["selected"] == 1, "feature name"].tolist()
        if len(selected) == 0:
            top_k = 5
            cand = significance_df[significance_df["t-value"] > 0].sort_values("t-value", ascending=False)
            selected = cand["feature name"].head(top_k).tolist()
            print(f"[WARN] No feature selected with p<{0.05}. Fallback to top-{top_k} by t-value: {selected}")

        # exclusion of not chosen features
        Xtr_sel = dataframes['X_train'][selected].copy()
        Xte_sel = dataframes['X_test'][selected].copy()
        ytr = Y_for_shap.loc[Xtr_sel.index]
        yte = Y_for_shap.loc[Xte_sel.index]

        # retrain with selected features
        model_reduced = clone(model)
        model_reduced.fit(Xtr_sel, ytr)

        print(f"\n[RE-TRAIN] {model_name} with {len(selected)} feature selected by SHAP")
        print("selected features:", selected)
        evaluate_and_report(
            model_name=f"{model_name}-selected",
            model=model_reduced,
            X_train=Xtr_sel, y_train=ytr,
            X_test=Xte_sel, y_test=yte,
            average="binary"
        )
        print("-" * 60)
