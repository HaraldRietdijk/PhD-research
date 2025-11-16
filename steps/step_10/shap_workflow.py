"""
shap_workflow.py

Core utilities to run SHAP-Select–based feature selection for a single model.

This module:

  • wraps the `shap_select` library call,
  • implements a robust fallback when no feature passes the p-value threshold,
  • retrains classifiers through the existing `classifier_selection` pipeline
    to keep training logic consistent,
  • evaluates reduced models and logs metrics,
  • collects per-feature p-values across seeds for global summaries,
  • returns a per-model grid-search DataFrame (threshold → performance).

It is used inside the step_10 main script to perform SHAP-Select grid searches
for each classifier and each random seed.
"""

import os
from typing import List

import numpy as np
import pandas as pd

from steps.step_10.step_10_selection_classifier import classifier_selection
from steps.step_10.metrics_utils import evaluate_and_report

# Try local import first; fall back to a relative path if the package is not installed.
try:
    from shap_select import shap_select
except ModuleNotFoundError:
    import sys
    import os as _os

    root = _os.path.realpath("..")
    sys.path.append(root)
    from shap_select import shap_select


# ---------------------------------------------------------------------------
# Feature fallback when none pass the p-threshold
# ---------------------------------------------------------------------------

def fallback_select_features(significance_df: pd.DataFrame, top_k: int = 5) -> List[str]:
    """Robust fallback strategy when no feature passes the SHAP p-value threshold.

    SHAP-Select can, in some rare cases, fail to select any feature if the
    threshold is too strict. This helper tries to recover a reasonable subset
    using the statistics available in `significance_df`:

      1) Prefer features with positive t-values (strong positive association),
         sorted by descending t-value.
      2) If none are positive, pick features with the largest absolute t-value
         (strongest effect regardless of sign).
      3) As a final fallback, pick features with the smallest p-values
         (`stat.significance`), even if above the threshold.

    If required columns are missing, it returns an empty list.

    Args:
        significance_df: DataFrame returned by `shap_select`, containing at
            least columns `t-value`, `stat.significance`, and `feature name`.
        top_k: Maximum number of features to return.

    Returns:
        List of feature names selected by the fallback strategy. May be empty.
    """
    sdf = significance_df.copy()

    # If required columns are missing, we cannot apply the fallback.
    if "t-value" not in sdf.columns or "stat.significance" not in sdf.columns:
        return []

    # 1) Try features with strictly positive t-values (strong positive signal).
    pos = sdf[np.isfinite(sdf["t-value"]) & (sdf["t-value"] > 0)]
    cand = (
        pos.sort_values("t-value", ascending=False)["feature name"]
        .head(top_k)
        .tolist()
    )
    if cand:
        return cand

    # 2) If no positive t-values, use largest absolute t-values (strong effect).
    abs_t = sdf[np.isfinite(sdf["t-value"])].copy()
    abs_t["abs_t"] = abs_t["t-value"].abs()
    cand = (
        abs_t.sort_values("abs_t", ascending=False)["feature name"]
        .head(top_k)
        .tolist()
    )
    if cand:
        return cand

    # 3) As a last resort, pick features with smallest p-values.
    pv = sdf[np.isfinite(sdf["stat.significance"])].copy()
    return (
        pv.sort_values("stat.significance", ascending=True)["feature name"]
        .head(top_k)
        .tolist()
    )


# ---------------------------------------------------------------------------
# Retraining helper (ALWAYS via classifier_selection)
# ---------------------------------------------------------------------------

def train_model_via_classifier_selection(
    app,
    base_folder: str,
    dataframes: dict,
    selected_features: List[str] | None,
    num_of_classes: int,
    model_name: str,
):
    """Retrain a single classifier through `classifier_selection` on filtered features.

    This function enforces a single training pipeline for both baseline and
    SHAP-selected models by delegating to `classifier_selection`. It optionally
    subsets the feature space to `selected_features`.

    Args:
        app: Application/context object, passed through to `classifier_selection`.
        base_folder: Base directory where retrained models will be stored.
        dataframes: Dictionary of DataFrames; must include 'X', 'X_train', 'X_test'.
        selected_features: List of feature names to keep. If None, all features
            are used (i.e., baseline training).
        num_of_classes: Number of target classes (e.g., 2 for binary).
        model_name: Name of the model to retrieve from the `classifier_selection`
            output (e.g., "RandomForestClassifier").

    Returns:
        The retrained model instance corresponding to `model_name`.

    Raises:
        KeyError: If `model_name` is not found in the `classifier_selection` result.
    """
    # We create a shallow copy of the dataframes dict, then optionally
    # subset the feature matrices.
    df2 = dict(dataframes)
    if selected_features is not None:
        df2 = df2.copy()
        df2["X"] = dataframes["X"][selected_features].copy()
        df2["X_train"] = dataframes["X_train"][selected_features].copy()
        df2["X_test"] = dataframes["X_test"][selected_features].copy()

    retrain_folder = os.path.join(base_folder, "classifiers_retrain")
    fitted = classifier_selection(app, retrain_folder, df2, num_of_classes)

    if model_name not in fitted:
        raise KeyError(f"Model '{model_name}' not found in classifier_selection output.")

    return fitted[model_name]


# ---------------------------------------------------------------------------
# Main SHAP grid-search routine for a single model
# ---------------------------------------------------------------------------

def run_shap_gridsearch(
    *,
    app,
    base_folder: str,
    dataframes: dict,
    y_bin,
    features: list,
    task: str,
    thresholds: list,
    seed: int,
    reporter_obj,
    model_name: str,
    baseline_model,
    num_of_classes: int,
    all_feature_records: list,
) -> pd.DataFrame:
    """Run SHAP-Select grid search for a single baseline model and log results.

    For the given `baseline_model`, this routine:

      1) Runs SHAP-Select for each threshold in `thresholds`.
      2) Reads per-feature significance statistics from `significance_df`.
      3) Uses SHAP-Select's `selected` column (1/0/-1) to pick features.
      4) If no feature passes the threshold, calls `fallback_select_features`.
      5) Retrains the classifier via `classifier_selection` on the selected features.
      6) Evaluates the reduced model on train/test using `evaluate_and_report`.
      7) Logs test metrics into `reporter_obj` with variant `shap@<threshold>`.
      8) Appends selected features and their p-values into `all_feature_records`
         for later global summarization.
      9) Collects a per-threshold row into a grid-search table.

    Args:
        app: Application/context object for downstream utilities.
        base_folder: Base path where classifier artifacts are stored.
        dataframes: Dictionary of DataFrames (X, X_train, X_test, etc.).
        y_bin: Binary target vector aligned with dataframes['X'] indices.
        features: List of feature names corresponding to columns in X.
        task: Task type passed to `shap_select` (e.g., "binary").
        thresholds: List of SHAP-Select thresholds to evaluate.
        seed: Random seed (for logging/traceability).
        reporter_obj: MetricsReporter-like object with a `.log(...)` method.
        model_name: Identifier of the classifier (consistent with baseline).
        baseline_model: Trained baseline model used as the SHAP explainer target.
        num_of_classes: Number of target classes.
        all_feature_records: List that gets extended with per-feature p-values
            for globally summarizing feature selection behavior.

    Returns:
        A DataFrame with one row per (threshold, seed) for this model, including:
            seed, model, threshold, n_selected, accuracy, f1, precision, recall, auc
    """
    gs_rows = []

    for thr_sel in thresholds:
        print(f"   -> threshold={thr_sel}")

        # Run SHAP-Select on the baseline model. We request extended outputs:
        #   significance_df: per-feature statistics (p-value, t-value, etc.)
        #   shap_values_df: full SHAP values per sample/feature (not used here)
        results = shap_select(
            tree_model=baseline_model,
            validation_df=dataframes["X"],
            target=y_bin,
            feature_names=features,
            task=task,
            threshold=thr_sel,
            return_extended_data=True,
        )
        significance_df, shap_values_df = results

        # Decode selection flags:
        #   selected ==  1 → feature selected
        #   selected ==  0 → discarded
        #   selected == -1 → selected but with negative coefficient
        selected = significance_df.loc[
            significance_df["selected"] == 1, "feature name"
        ].tolist()
        dropped = significance_df.loc[
            significance_df["selected"] == 0, "feature name"
        ].tolist()
        neg_coef = significance_df.loc[
            significance_df["selected"] == -1, "feature name"
        ].tolist()

        print(
            f"      selected={len(selected)} | "
            f"discarded={len(dropped)} | negative={len(neg_coef)}"
        )

        # If no feature is selected, attempt a more lenient fallback.
        if len(selected) == 0:
            selected = fallback_select_features(significance_df, top_k=5)
            print(
                f"      [WARN] No feature selected with p<{thr_sel}. "
                f"Fallback: {selected}"
            )

        # Still no features? Log NaNs for this threshold and move on.
        if len(selected) == 0:
            reporter_obj.log(
                model=model_name,
                variant=f"shap@{thr_sel}",
                n_features=0,
                metrics={
                    "accuracy": np.nan,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "auc": np.nan,
                    "cm": None,
                    "y_pred": None,
                    "scores": None,
                },
            )
            gs_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "threshold": float(thr_sel),
                    "n_selected": 0,
                    "accuracy": np.nan,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "auc": np.nan,
                }
            )
            continue

        # -------------------------------------------------------------------
        # Retrain strictly via classifier_selection on the selected features
        # -------------------------------------------------------------------
        model_reduced = train_model_via_classifier_selection(
            app=app,
            base_folder=base_folder,
            dataframes=dataframes,
            selected_features=selected,
            num_of_classes=num_of_classes,
            model_name=model_name,
        )

        # Build reduced train/test matrices.
        Xtr_sel = dataframes["X_train"][selected].copy()
        Xte_sel = dataframes["X_test"][selected].copy()
        ytr = y_bin.loc[Xtr_sel.index]
        yte = y_bin.loc[Xte_sel.index]

        # Evaluate the reduced model and print metrics block.
        mt_tr2, mt_te2 = evaluate_and_report(
            model_name=f"{model_name}-selected@{thr_sel}",
            model=model_reduced,
            X_train=Xtr_sel,
            y_train=ytr,
            X_test=Xte_sel,
            y_test=yte,
            average="binary",
        )

        # Log test metrics into the per-seed reporter.
        reporter_obj.log(
            model=model_name,
            variant=f"shap@{thr_sel}",
            n_features=len(selected),
            metrics=mt_te2,
        )

        # Collect per-feature p-values for globally summarizing selection behavior.
        sel_rows = significance_df.loc[
            significance_df["selected"] == 1,
            ["feature name", "stat.significance"],
        ]
        for _, r in sel_rows.iterrows():
            all_feature_records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "feature": r["feature name"],
                    "p_value": float(r["stat.significance"]),
                }
            )

        # Append one grid-search row for this threshold.
        gs_rows.append(
            {
                "seed": seed,
                "model": model_name,
                "threshold": float(thr_sel),
                "n_selected": int(len(selected)),
                "accuracy": float(mt_te2["accuracy"]),
                "f1": float(mt_te2["f1"]),
                "precision": float(mt_te2["precision"]),
                "recall": float(mt_te2["recall"]),
                "auc": (np.nan if mt_te2["auc"] is None else float(mt_te2["auc"])),
            }
        )

    return pd.DataFrame(gs_rows)
