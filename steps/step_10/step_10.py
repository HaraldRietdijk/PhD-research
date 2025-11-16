import os
import sys
import random
import numpy as np
import pandas as pd
import glob
import math

from steps.step_10.reporter import MetricsReporter
from steps.step_generic_code.dataframe_knee_operations import get_dataframe
from steps.step_generic_code.general_functions import start_logging
from steps.step_10.step_10_selection_classifier import classifier_selection
from steps.step_10.step_10_plot import plot_results

from steps.step_10.metrics_utils import evaluate_and_report
from steps.step_10.report_utils import (
    load_all_seed_metrics, write_feature_summary_tables,
    plot_aggregate_bars, plot_per_model_pairwise_bars, build_aggregate_pdf
)
from steps.step_10.shap_workflow import run_shap_gridsearch

# Seeds and SHAP thresholds
SEEDS = [10, 222, 360, 420, 576, 66, 777, 82, 9000, 1234]
THRESHOLDS = [0.01, 0.04, 0.07, 0.12, 0.15]

def do_step_10(app):
    # Global settings
    num_of_classes = 2
    base_folder = 'results/fs_shap'
    start_logging(base_folder)

    # Collected across seeds → global summary
    all_feature_records = []

    for seed in SEEDS:
        np.random.seed(seed); random.seed(seed)
        print("\n" + "="*70)
        print(f"[RUN] Seed = {seed}")
        print("="*70)

        # Data
        dataframes = get_dataframe(app, num_of_classes)
        features = dataframes['X'].columns.tolist()

        # Baseline training via classifier_selection
        folder_clf = os.path.join(base_folder, 'classifiers')
        fitted_models = classifier_selection(app, folder_clf, dataframes, num_of_classes)
        plot_results(app, folder_clf, num_of_classes)

        # Binary target for SHAP
        y_cont = dataframes['Y']
        thr = y_cont.mean()
        y_bin = (y_cont >= thr).astype(int)
        print("Mean", float(thr))
        task = "binary"

        # Per-seed report dir
        report_dir = os.path.join(base_folder, "report", f"seed_{seed}")
        os.makedirs(report_dir, exist_ok=True)
        reporter_obj = MetricsReporter(report_dir)

        print("Fitted models:", fitted_models)

        # Baseline metrics logging
        for model_name, model in fitted_models.items():
            mt_tr, mt_te = evaluate_and_report(
                model_name=model_name,
                model=model,
                X_train=dataframes['X_train'],
                y_train=y_bin.loc[dataframes['X_train'].index],
                X_test=dataframes['X_test'],
                y_test=y_bin.loc[dataframes['X_test'].index],
                average="binary"
            )
            reporter_obj.log(
                model=model_name, variant="base",
                n_features=len(dataframes['X_train'].columns), metrics=mt_te
            )

        # SHAP grid-search per model (retrain always through classifier_selection)
        for model_name, baseline_model in fitted_models.items():
            print(f"[SHAP-SELECT] Grid-search on {model_name} ({type(baseline_model).__name__})")
            gs_df = run_shap_gridsearch(
                app=app, base_folder=base_folder, dataframes=dataframes, y_bin=y_bin,
                features=features, task=task, thresholds=THRESHOLDS, seed=seed,
                reporter_obj=reporter_obj, model_name=model_name,
                baseline_model=baseline_model, num_of_classes=num_of_classes,
                all_feature_records=all_feature_records
            )
            gs_df.to_csv(os.path.join(report_dir, f"gridsearch_thresholds_{model_name}.csv"), index=False)

        # Per-seed artifacts
        reporter_obj.save_csv(filename="metrics_report.csv")
        reporter_obj.plot_baseline_vs_shap_bars(
            filename="baseline_vs_shap_per_model.png",
            title=f"Baseline vs SHAP-selected (seed={seed})"
        )

        seed_gs_paths = sorted(glob.glob(os.path.join(report_dir, "gridsearch_thresholds_*.csv")))
        if seed_gs_paths:
            _dfs = [pd.read_csv(p) for p in seed_gs_paths]
            _all = pd.concat(_dfs, ignore_index=True)
            _all.to_csv(os.path.join(report_dir, "gridsearch_thresholds.csv"), index=False)

        print(f"[REPORT] Saved in: {report_dir}")

    # ===== Global aggregate report =====
    df_metrics = load_all_seed_metrics(base_folder)
    out_agg_dir = os.path.join(base_folder, "report_aggregate")
    os.makedirs(out_agg_dir, exist_ok=True)

    # Feature selection summary (frequency + mean p-value)
    feature_summary = write_feature_summary_tables(all_feature_records, out_dir=out_agg_dir)

    # Charts & PDF
    plot_aggregate_bars(df_metrics=df_metrics, out_dir=out_agg_dir)
    plot_per_model_pairwise_bars(df_metrics=df_metrics, out_dir=out_agg_dir)
    build_aggregate_pdf(base_folder=base_folder, df_metrics=df_metrics, feature_summary=feature_summary)

    print(f"[AGG] Figures and PDF in: {out_agg_dir}")
