"""
report_utils.py

Utilities for aggregating and visualizing results of the SHAP-Select
experiments across multiple seeds and classifiers.

This module provides:

  • Aggregation helpers:
      - loading all per-seed metrics
      - computing 95% confidence intervals

  • Feature-selection summaries:
      - frequency of selection per model × feature
      - mean p-value per model × feature
      - export to CSV and insertion into PDF

  • Plotting utilities:
      - aggregate bar plots (baseline vs SHAP, mean ± 95% CI)
      - per-model bar plots (baseline vs SHAP across metrics)

  • PDF builder:
      - a single aggregate PDF combining metric plots and feature tables

Backward-compatible aliases (`_ci95`, `_load_all_seed_metrics`, etc.) are kept
so that older code can still import these functions using the previous names.
"""

import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def ci95(series: pd.Series) -> float:
    """Compute the half-width of a 95% confidence interval (normal approx).

    This is a general-purpose helper used to attach error bars to mean
    performance metrics across seeds.

    Args:
        series: Pandas Series of numeric values (e.g., accuracies over seeds).

    Returns:
        Half-width of the 95% confidence interval:
            1.96 * std(sample) / sqrt(n)
        Returns NaN if the series has fewer than 2 valid observations.
    """
    x = series.dropna().astype(float)
    n = len(x)
    if n <= 1:
        return np.nan
    return 1.96 * x.std(ddof=1) / math.sqrt(n)


# Backward-compat alias (for legacy code that used `_ci95`)
_ci95 = ci95


def load_all_seed_metrics(base_folder: str) -> pd.DataFrame:
    """Load all per-seed `metrics_report.csv` files into a single DataFrame.

    Expected directory layout:
        base_folder/
          report/
            seed_10/metrics_report.csv
            seed_222/metrics_report.csv
            ...

    Each file is assumed to contain metrics for:
      - baseline models (variant == "base")
      - SHAP-selected models (variant == "shap" or similar)

    A `seed` column is added, parsed from the folder name.

    Args:
        base_folder: Root directory that contains the `report/seed_*` folders.

    Returns:
        A concatenated DataFrame with metrics from all seeds.

    Raises:
        FileNotFoundError: If no `metrics_report.csv` files are found.
    """
    paths = sorted(glob.glob(os.path.join(base_folder, "report", "seed_*", "metrics_report.csv")))
    frames = []
    for p in paths:
        # Extract the seed id from the folder name, e.g. "seed_10" -> 10
        try:
            seed = int(os.path.basename(os.path.dirname(p)).split("_")[-1])
        except Exception:
            seed = None
        df = pd.read_csv(p)
        df["seed"] = seed
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No per-seed metrics_report.csv found.")

    return pd.concat(frames, ignore_index=True)


# Backward-compat alias
_load_all_seed_metrics = load_all_seed_metrics


# ---------------------------------------------------------------------------
# Feature summary (frequency & mean p-value)
# ---------------------------------------------------------------------------

def write_feature_summary_tables(all_feature_records: list, out_dir: str) -> pd.DataFrame:
    """Build feature-selection frequency and mean p-value tables per model × feature.

    This function expects a list of records collected during the SHAP-Select
    runs. Each record typically has the form:
        {
            "seed": <int>,
            "model": <str>,
            "feature": <str>,
            "p_value": <float>,
            ...
        }

    It then aggregates these records to compute, for each (model, feature):

      • times_selected: how many times the feature was selected across all seeds
      • mean_p_value: the average p-value when the feature was selected

    The resulting summary is saved as:
        <out_dir>/feature_selection_summary.csv

    Args:
        all_feature_records: List of dict-like records logged during runs.
        out_dir: Output directory for the summary CSV.

    Returns:
        A DataFrame with columns:
            model, feature, times_selected, mean_p_value

    Raises:
        RuntimeError: If no feature records were provided.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not all_feature_records:
        raise RuntimeError(
            "No feature records collected. Make sure you append them during the run."
        )

    # Convert list of records into a DataFrame.
    # Typical columns: seed, model, feature, p_value, ...
    df = pd.DataFrame(all_feature_records)

    # Count how many times each feature is selected for each model.
    freq = (
        df.groupby(["model", "feature"])
        .size()
        .rename("times_selected")
        .reset_index()
    )

    # Compute mean p-value per (model, feature).
    mean_p = (
        df.groupby(["model", "feature"])["p_value"]
        .mean()
        .rename("mean_p_value")
        .reset_index()
    )

    # Merge frequency and mean p-value, then sort:
    #  - by model
    #  - then by descending frequency
    #  - then by ascending mean p-value (best first)
    summary = (
        freq.merge(mean_p, on=["model", "feature"])
        .sort_values(
            ["model", "times_selected", "mean_p_value"],
            ascending=[True, False, True],
        )
    )

    out_csv = os.path.join(out_dir, "feature_selection_summary.csv")
    summary.to_csv(out_csv, index=False)
    return summary


# Backward-compat alias
_write_feature_summary_tables = write_feature_summary_tables


def _add_feature_tables_to_pdf(
    summary: pd.DataFrame,
    pdf: PdfPages,
    max_rows_per_page: int = 30,
) -> None:
    """Add per-model feature-selection tables into the PDF.

    Each model gets one or more pages, depending on how many features were
    selected. For each model, features are sorted by:

        1. times_selected (descending)
        2. mean_p_value (ascending)

    Args:
        summary: Output DataFrame from `write_feature_summary_tables`.
        pdf: An open PdfPages instance where pages will be appended.
        max_rows_per_page: Max number of feature rows per page.
    """
    for model in summary["model"].unique():
        sub = summary[summary["model"] == model].copy()
        if sub.empty:
            continue

        sub = sub.sort_values(
            ["times_selected", "mean_p_value"],
            ascending=[False, True],
        )

        # Split into chunks so that tables do not become too dense.
        chunks = [
            sub.iloc[i : i + max_rows_per_page]
            for i in range(0, len(sub), max_rows_per_page)
        ]

        for chunk in chunks:
            fig, ax = plt.subplots(figsize=(10, 0.35 * len(chunk) + 1.5))
            ax.axis("off")
            ax.set_title(
                f"{model} — Selected features (freq & mean p-value)",
                loc="left",
                fontsize=12,
                pad=10,
            )

            # Header row + table content
            table_data = [["feature", "times_selected", "mean_p_value"]]
            for _, r in chunk.iterrows():
                table_data.append(
                    [
                        r["feature"],
                        int(r["times_selected"]),
                        f"{r['mean_p_value']:.4g}",
                    ]
                )

            table = ax.table(
                cellText=table_data,
                loc="center",
                cellLoc="left",
                colLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_aggregate_bars(df_metrics: pd.DataFrame, out_dir: str) -> None:
    """Plot aggregate bar charts (base vs shap) for each metric.

    For each metric (accuracy, F1, precision, recall, AUROC), this function:
      - computes mean and 95% CI across seeds for each (model, variant),
      - plots side-by-side bars for baseline ("base") and SHAP ("shap"),
        with error bars representing the 95% CI,
      - saves each plot as:
            <out_dir>/aggregate_<metric>.png

    Args:
        df_metrics: DataFrame with columns at least:
            model, variant, accuracy, f1, precision, recall, auc, seed
        out_dir: Output directory for the PNG files.
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]

    agg = (
        df_metrics.groupby(["model", "variant"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_ci95=("accuracy", ci95),
            f1_mean=("f1", "mean"),
            f1_ci95=("f1", ci95),
            precision_mean=("precision", "mean"),
            precision_ci95=("precision", ci95),
            recall_mean=("recall", "mean"),
            recall_ci95=("recall", ci95),
            auc_mean=("auc", "mean"),
            auc_ci95=("auc", ci95),
        )
        .reset_index()
    )

    for m in metrics:
        base = agg[agg["variant"] == "base"].set_index("model")
        shap_ = agg[agg["variant"] == "shap"].set_index("model")
        common = base.index.intersection(shap_.index)
        if len(common) == 0:
            continue

        labels = list(common)
        base_mean = base.loc[common, f"{m}_mean"].values
        base_ci   = base.loc[common, f"{m}_ci95"].values
        shap_mean = shap_.loc[common, f"{m}_mean"].values
        shap_ci   = shap_.loc[common, f"{m}_ci95"].values

        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(
            x - width / 2,
            base_mean,
            width,
            yerr=base_ci,
            capsize=4,
            label="base",
        )
        ax.bar(
            x + width / 2,
            shap_mean,
            width,
            yerr=shap_ci,
            capsize=4,
            label="shap",
        )
        ax.set_title(f"{m}: mean ± 95% CI (across seeds)")
        ax.set_ylabel(m)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend()
        plt.tight_layout()

        figpath = os.path.join(out_dir, f"aggregate_{m}.png")
        plt.savefig(figpath, dpi=160)
        plt.close()


# Backward-compat alias
_plot_aggregate_bars = plot_aggregate_bars


def plot_per_model_pairwise_bars(df_metrics: pd.DataFrame, out_dir: str) -> None:
    """Plot baseline vs shap comparison for each model across all metrics.

    For each model, this function creates a bar plot where:
      - x-axis: metrics (accuracy, F1, precision, recall, AUROC)
      - bars: baseline ("base") and SHAP ("shap") mean scores
      - error bars: 95% CI across seeds

    Each plot is saved as:
        <out_dir>/per_model_<model>.png

    Args:
        df_metrics: DataFrame with per-seed metrics.
        out_dir: Directory for the output PNG files.
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]

    agg = (
        df_metrics.groupby(["model", "variant"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_ci95=("accuracy", ci95),
            f1_mean=("f1", "mean"),
            f1_ci95=("f1", ci95),
            precision_mean=("precision", "mean"),
            precision_ci95=("precision", ci95),
            recall_mean=("recall", "mean"),
            recall_ci95=("recall", ci95),
            auc_mean=("auc", "mean"),
            auc_ci95=("auc", ci95),
        )
        .reset_index()
    )

    models = sorted(agg["model"].unique())
    for mdl in models:
        sub = agg[agg["model"] == mdl]
        # Require both base and shap variants for a meaningful comparison
        if set(sub["variant"]) & {"base", "shap"} != {"base", "shap"}:
            continue

        base = sub[sub["variant"] == "base"].iloc[0]
        shap_ = sub[sub["variant"] == "shap"].iloc[0]

        base_means = np.array([base[f"{m}_mean"] for m in metrics], dtype=float)
        base_cis   = np.array([base[f"{m}_ci95"] for m in metrics], dtype=float)
        shap_means = np.array([shap_[f"{m}_mean"] for m in metrics], dtype=float)
        shap_cis   = np.array([shap_[f"{m}_ci95"] for m in metrics], dtype=float)

        x = np.arange(len(metrics))
        width = 0.38
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(
            x - width / 2,
            base_means,
            width,
            yerr=base_cis,
            capsize=4,
            label="base",
        )
        ax.bar(
            x + width / 2,
            shap_means,
            width,
            yerr=shap_cis,
            capsize=4,
            label="shap",
        )
        ax.set_title(f"{mdl} — mean ± 95% CI across seeds")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=0, ha="center")
        ax.set_ylabel("score")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"per_model_{mdl}.png")
        plt.savefig(out_path, dpi=160)
        plt.close()


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def build_aggregate_pdf(
    base_folder: str,
    df_metrics: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> str:
    """Create a comprehensive aggregate PDF report.

    The PDF includes:

      1) Aggregate metric bars (one page per metric):
         - baseline vs SHAP mean scores
         - 95% CIs across seeds

      2) Per-model pairwise bars (one page per model):
         - baseline vs SHAP across all metrics

      3) Per-model feature-selection tables:
         - frequency of selection (times_selected)
         - mean p-value (mean_p_value)
         - split across pages if necessary

    Args:
        base_folder: Base directory where `report_aggregate/` will be created.
        df_metrics: Per-seed metrics DataFrame (baseline + SHAP).
        feature_summary: Summary DataFrame from `write_feature_summary_tables`.

    Returns:
        The full path to the generated PDF file.
    """
    out_dir = os.path.join(base_folder, "report_aggregate")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "aggregate_report.pdf")

    metrics = ["accuracy", "f1", "precision", "recall", "auc"]
    agg = (
        df_metrics.groupby(["model", "variant"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_ci95=("accuracy", ci95),
            f1_mean=("f1", "mean"),
            f1_ci95=("f1", ci95),
            precision_mean=("precision", "mean"),
            precision_ci95=("precision", ci95),
            recall_mean=("recall", "mean"),
            recall_ci95=("recall", ci95),
            auc_mean=("auc", "mean"),
            auc_ci95=("auc", ci95),
        )
        .reset_index()
    )

    with PdfPages(pdf_path) as pdf:
        # 1) Aggregate bars (one page per metric)
        for m in metrics:
            base = agg[agg["variant"] == "base"].set_index("model")
            shap_ = agg[agg["variant"] == "shap"].set_index("model")
            common = base.index.intersection(shap_.index)
            if len(common) == 0:
                continue

            labels = list(common)
            base_mean = base.loc[common, f"{m}_mean"].values
            base_ci   = base.loc[common, f"{m}_ci95"].values
            shap_mean = shap_.loc[common, f"{m}_mean"].values
            shap_ci   = shap_.loc[common, f"{m}_ci95"].values

            x = np.arange(len(labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.bar(
                x - width / 2,
                base_mean,
                width,
                yerr=base_ci,
                capsize=4,
                label="base",
            )
            ax.bar(
                x + width / 2,
                shap_mean,
                width,
                yerr=shap_ci,
                capsize=4,
                label="shap",
            )
            ax.set_title(f"{m}: mean ± 95% CI (across seeds)")
            ax.set_ylabel(m)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 2) Per-model pairwise bars (one page per model)
        models = sorted(agg["model"].unique())
        for mdl in models:
            sub = agg[agg["model"] == mdl]
            if set(sub["variant"]) & {"base", "shap"} != {"base", "shap"}:
                continue

            base_row = sub[sub["variant"] == "base"].iloc[0]
            shap_row = sub[sub["variant"] == "shap"].iloc[0]

            base_means = np.array([base_row[f"{m}_mean"] for m in metrics], dtype=float)
            base_cis   = np.array([base_row[f"{m}_ci95"] for m in metrics], dtype=float)
            shap_means = np.array([shap_row[f"{m}_mean"] for m in metrics], dtype=float)
            shap_cis   = np.array([shap_row[f"{m}_ci95"] for m in metrics], dtype=float)

            x = np.arange(len(metrics))
            width = 0.38
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.bar(
                x - width / 2,
                base_means,
                width,
                yerr=base_cis,
                capsize=4,
                label="base",
            )
            ax.bar(
                x + width / 2,
                shap_means,
                width,
                yerr=shap_cis,
                capsize=4,
                label="shap",
            )
            ax.set_title(f"{mdl}: mean ± 95% CI across metrics")
            ax.set_ylabel("score")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=0, ha="center")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 3) Feature tables (one or more pages per model)
        _add_feature_tables_to_pdf(feature_summary, pdf)

    print(f"[AGG] PDF saved: {pdf_path}")
    return pdf_path


# Backward-compat alias
_build_aggregate_pdf = build_aggregate_pdf
