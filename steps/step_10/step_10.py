import os
import sys
from steps.step_10.reporter import MetricsReporter, MetricsEntry
from steps.step_10.aggregate_report import (
    _ci95, _load_all_seed_metrics, _plot_aggregate_bars, _write_feature_summary_tables, _add_feature_tables_to_pdf, _build_aggregate_pdf
)

from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging

from steps.step_10.step_10_selection_classifier import classifier_selection
from steps.step_generic_code.general_variables.general_variables_all_shap import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS
from steps.step_10.step_10_generic_selection import save_pickle_and_metrics
from steps.step_10.step_10_plot import plot_results

import shap

import numpy as np
import pandas as pd
import random
import math
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


# Seeds for a better generalization of results
SEEDS = [10, 270, 333, 41, 500, 999, 123, 456, 789, 888] 


# Check if the model is a tree-based model (since SHAP TreeExplainer works only for them)
def is_tree_model(model) -> bool:
    """Keep for potential future filtering (currently not used to skip)."""
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)):
        return True
    if isinstance(model, AdaBoostClassifier):
        base = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
        return (base is None) or isinstance(base, (DecisionTreeClassifier, DecisionTreeRegressor))
    return False


def _decision_scores_or_proba(model, X):
    """Return a continuous score: predict_proba[:,1] if available,
    otherwise decision_function, otherwise predictions (fallback)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2:
            # binary: use positive class column if available
            if proba.shape[1] == 2:
                return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    # fallback (discrete preds)
    return model.predict(X)


def compute_metrics(model, X, y, average: str = "binary") -> dict:
    """Compute standard classification metrics (and store y_pred / scores)."""
    y_pred = model.predict(X)
    scores = _decision_scores_or_proba(model, X)

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average=average, zero_division=0)
    rec  = recall_score(y, y_pred, average=average, zero_division=0)
    f1   = f1_score(y, y_pred, average=average, zero_division=0)
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

# manage AUC formatting when None
def _fmt_auc(x):
    return f"{x:.4f}" if (x is not None) else "n/a"


def print_metrics_block(model_name: str, mt_train: dict, mt_test: dict) -> None:
    """Shows metrics in a compact block."""
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
    """Wrapper to compute + print metrics for train and test."""
    mt_train = compute_metrics(model, X_train, y_train, average=average)
    mt_test  = compute_metrics(model, X_test,  y_test,  average=average)
    print_metrics_block(model_name, mt_train, mt_test)
    return mt_train, mt_test



def do_step_10(app):
    # Global settings
    num_of_classes = 2
    base_folder = 'results/fs_shap'
    start_logging(base_folder)

    # Collect feature selections across seeds to build global summary later
    all_feature_records = []  # dicts: {seed, model, feature, p_value}

    for seed in SEEDS:
        
        np.random.seed(seed)
        random.seed(seed)

        print("\n" + "="*70)
        print(f"[RUN] Seed = {seed}")
        print("="*70)

        # Prepare data
        dataframes = get_dataframe(app, num_of_classes)
        features = dataframes['X'].columns.tolist()

        # Train classifiers and plot base curves
        folder_clf = os.path.join(base_folder, 'classifiers')
        fitted_models = classifier_selection(app, folder_clf, dataframes, num_of_classes)
        plot_results(app, folder_clf, num_of_classes)

        # Binarize target for SHAP (mean threshold)
        y_cont = dataframes['Y']
        thr = y_cont.mean()
        y_bin = (y_cont >= thr).astype(int)
        print("Mean", float(thr))
        Y_for_shap = y_bin
        task = "binary"

        # Results folders (per-seed)
        shap_folder = os.path.join(base_folder, 'shap_select', f"seed_{seed}")
        os.makedirs(shap_folder, exist_ok=True)

        # Reporter (per-seed)
        report_dir = os.path.join(base_folder, "report", f"seed_{seed}")
        reporter_obj = MetricsReporter(report_dir)

        print("Fitted models:", fitted_models)

        for model_name, model in fitted_models.items():
            # compute metrics
            mt_tr, mt_te = evaluate_and_report(
                model_name=model_name,
                model=model,
                X_train=dataframes['X_train'],
                y_train=Y_for_shap.loc[dataframes['X_train'].index],  # align by index
                X_test=dataframes['X_test'],
                y_test=Y_for_shap.loc[dataframes['X_test'].index],    # align by index
                average="binary"
            )

            # Log metrics of the base model to the per-seed reporter
            reporter_obj.log(
                model=model_name,
                variant="base",
                n_features=len(dataframes['X_train'].columns),
                metrics=mt_te
            )

            # shap select 
            model_type = type(model).__name__
            print(f"[SHAP-SELECT] Execute on {model_name} ({model_type})")

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

            # Selected / dropped / negative
            selected = significance_df.loc[significance_df["selected"] == 1, "feature name"].tolist()
            dropped  = significance_df.loc[significance_df["selected"] == 0, "feature name"].tolist()
            neg_coef = significance_df.loc[significance_df["selected"] == -1, "feature name"].tolist()

            print(f"\nResults for {model_name}:")
            print(" Chosen:", selected)
            print(" Discarded:", dropped)
            if neg_coef:
                print(" Negative Coefficients:", neg_coef)
            print("-" * 60)

            # Fallback if none pass threshold (keep the behavior)
            selected = significance_df.loc[significance_df["selected"] == 1, "feature name"].tolist()
            if len(selected) == 0:
                top_k = 5
                cand = significance_df[significance_df["t-value"] > 0].sort_values("t-value", ascending=False)
                selected = cand["feature name"].head(top_k).tolist()
                print(f"[WARN] No feature selected with p<0.05. Fallback to top-{top_k} by t-value: {selected}")
                # If you don't want to evaluate a pseudo-selected set, continue
                continue

            # Retrain on selected features
            Xtr_sel = dataframes['X_train'][selected].copy()
            Xte_sel = dataframes['X_test'][selected].copy()
            ytr = Y_for_shap.loc[Xtr_sel.index]
            yte = Y_for_shap.loc[Xte_sel.index]

            model_reduced = clone(model)
            model_reduced.fit(Xtr_sel, ytr)

            print(f"\n[RE-TRAIN] {model_name} with {len(selected)} features selected by SHAP")
            print("selected features:", selected)
            mt_tr2, mt_te2 = evaluate_and_report(
                model_name=f"{model_name}-selected",
                model=model_reduced,
                X_train=Xtr_sel, y_train=ytr,
                X_test=Xte_sel, y_test=yte,
                average="binary"
            )

            # Log SHAP-variant to the per-seed reporter
            reporter_obj.log(
                model=model_name,
                variant="shap",
                n_features=len(selected),
                metrics=mt_te2
            )
            print("-" * 60)

            # collect feature selection records for global summary
            # Keeping also the p-value (significance)
            sel_rows = significance_df.loc[significance_df["selected"] == 1, ["feature name", "stat.significance"]]
            for _, r in sel_rows.iterrows():
                all_feature_records.append({
                    "seed": seed,
                    "model": model_name,
                    "feature": r["feature name"],
                    "p_value": float(r["stat.significance"])
                })

        # build per-seed report (CSV + bar plot)
        reporter_obj.save_csv(filename="metrics_report.csv")
        reporter_obj.plot_baseline_vs_shap_bars(
            filename="baseline_vs_shap_per_model.png",
            title=f"Baseline vs SHAP-selected (seed={seed})"
        )
        print(f"[REPORT] Saved in: {report_dir}")

    # Global aggregate report across seeds
    df_metrics = _load_all_seed_metrics(base_folder)
    out_agg_dir = os.path.join(base_folder, "report_aggregate")
    os.makedirs(out_agg_dir, exist_ok=True)

    # Save aggregate stats (mean/std/ci95 per model×variant)
    agg_stats = (df_metrics
                 .groupby(["model","variant"])
                 .agg(accuracy_mean=("accuracy","mean"), accuracy_std=("accuracy","std"), accuracy_n=("accuracy","count"),
                      f1_mean=("f1","mean"), f1_std=("f1","std"), f1_n=("f1","count"),
                      precision_mean=("precision","mean"), precision_std=("precision","std"), precision_n=("precision","count"),
                      recall_mean=("recall","mean"), recall_std=("recall","std"), recall_n=("recall","count"),
                      auc_mean=("auc","mean"), auc_std=("auc","std"), auc_n=("auc","count"))
                 .reset_index())

    # Add 95% CI columns
    for m in ["accuracy","f1","precision","recall","auc"]:
        ci = []
        for _, row in agg_stats.iterrows():
            n = row[f"{m}_n"]
            if n and n > 1:
                ci.append(1.96 * (row[f"{m}_std"] / math.sqrt(n)))
            else:
                ci.append(np.nan)
        agg_stats[f"{m}_ci95"] = ci

    agg_csv_path = os.path.join(out_agg_dir, "metrics_report_aggregate.csv")
    agg_stats.to_csv(agg_csv_path, index=False)

    # Save feature selection frequency + mean p-value CSV
    feature_summary = _write_feature_summary_tables(all_feature_records, out_dir=out_agg_dir)

    # Save aggregate metric bar PNGs
    _plot_aggregate_bars(df_metrics=df_metrics, out_dir=out_agg_dir)

    # Build a dedicated PDF with aggregate charts + feature tables
    _build_aggregate_pdf(base_folder=base_folder, df_metrics=df_metrics, feature_summary=feature_summary)

    print(f"[AGG] Aggregate CSV saved: {agg_csv_path}")
    print(f"[AGG] Feature summary saved: {os.path.join(out_agg_dir, 'feature_selection_summary.csv')}")
    print(f"[AGG] Figures and PDF in: {out_agg_dir}")
