"""
build_threshold_report.py

Standalone script that reads per-seed CSVs already produced by the SHAP-Select
pipeline and generates:

  - A per-model PDF report:
      * baseline vs SHAP "best-per-seed" histogram (including AUROC)
      * metric trends vs threshold (accuracy, F1, precision, recall)
      * number of selected features vs threshold

  - Two CSV summary tables:
      * best_per_seed_means.csv
          Mean performance at the best SHAP-Select threshold per seed, aggregated
          across seeds (one row per model).
      * best_overall_by_model.csv
          Mean and standard deviation across seeds of the "best-per-seed" runs,
          plus the average threshold and average number of features selected.

This script is **purely post-hoc**: it does not train models or run SHAP;
it only aggregates and visualizes metrics that are already stored on disk.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# ----------------------------
# Basic helpers
# ----------------------------

def _ci95(series: pd.Series) -> float:
    """Compute half-width of a 95% confidence interval using normal approximation.

    This helper is currently **not used** in the report, but kept as a utility
    in case CI-based error bars are needed in the future.

    Args:
        series: Pandas Series of numeric values.

    Returns:
        The half-width of the 95% CI. Returns NaN if there are not enough points.
    """
    x = series.dropna().astype(float)
    n = len(x)
    if n <= 1:
        return np.nan
    # Standard error = std / sqrt(n); 1.96 is the z-score for a 95% CI
    return 1.96 * x.std(ddof=1) / math.sqrt(n)


def load_all_seed_metrics(base_folder: str) -> pd.DataFrame:
    """Load per-seed `metrics_report.csv` files (baseline + shap@thr variants).

    Expected directory structure:
        base_folder/
          report/
            seed_10/metrics_report.csv
            seed_222/metrics_report.csv
            ...

    Each CSV is expected to contain both:
      - baseline rows (variant == "base")
      - SHAP-Select rows (variant like "shap@0.10", etc.)

    Args:
        base_folder: Base path that contains the `report/seed_*` subfolders.

    Returns:
        A concatenated DataFrame with an additional `seed` column inferred
        from the folder name (e.g. seed_10 -> seed=10).

    Raises:
        FileNotFoundError: If no metrics_report.csv files are found.
    """
    paths = sorted(glob.glob(os.path.join(base_folder, "report", "seed_*", "metrics_report.csv")))
    frames = []
    for p in paths:
        # Infer seed number from folder name, e.g. ".../seed_10/metrics_report.csv"
        try:
            seed = int(os.path.basename(os.path.dirname(p)).split("_")[-1])
        except Exception:
            seed = None
        df = pd.read_csv(p)
        df["seed"] = seed
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No per-seed metrics_report.csv found under report/seed_*/")

    return pd.concat(frames, ignore_index=True)


def load_all_gridsearch(base_folder: str) -> pd.DataFrame:
    """Load per-seed `gridsearch_thresholds.csv` files (merged across models).

    Each `gridsearch_thresholds.csv` should already contain grid-search results
    for multiple thresholds and models for a single seed.

    Args:
        base_folder: Base path that contains the `report/seed_*` subfolders.

    Returns:
        A concatenated DataFrame with all grid-search results across seeds.

    Raises:
        FileNotFoundError: If no gridsearch_thresholds.csv files are found.
    """
    paths = sorted(glob.glob(os.path.join(base_folder, "report", "seed_*", "gridsearch_thresholds.csv")))
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No gridsearch_thresholds.csv found under report/seed_*/")

    return pd.concat(frames, ignore_index=True)


# ----------------------------
# Core aggregations
# ----------------------------

def compute_baseline_means(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute mean baseline metrics (all features, no feature selection) per model.

    This function only considers rows with variant == "base" and averages metrics
    across seeds, resulting in a single row per model.

    Args:
        df_metrics: DataFrame containing metrics for multiple seeds and variants.

    Returns:
        DataFrame with columns:
            model, accuracy, f1, precision, recall, auc
        where values are means across seeds.
    """
    # Filter to baseline runs only (no feature selection)
    base = df_metrics[df_metrics["variant"] == "base"].copy()
    cols_keep = ["model", "seed", "accuracy", "f1", "precision", "recall", "auc"]
    base = base.loc[:, [c for c in cols_keep if c in base.columns]]

    # Aggregate across seeds, one row per model
    agg = (
        base.groupby("model")
        .agg(
            accuracy=("accuracy", "mean"),
            f1=("f1", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            auc=("auc", "mean"),
        )
        .reset_index()
    )
    return agg


def compute_best_per_seed_means(gs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean metrics of the best-per-seed SHAP selections for each model.

    For each (model, seed) combination, this function:
      1. Selects the row (threshold) with the highest accuracy.
      2. Aggregates those best rows across seeds, computing mean metrics.

    This is used for:
      - Producing a fair SHAP vs baseline comparison (mean across seeds).
      - Generating the "SHAP best" tables.

    Args:
        gs_df: Grid-search DataFrame across models, thresholds and seeds.

    Returns:
        DataFrame with one row per model, including:
            accuracy_mean, f1_mean, precision_mean, recall_mean, auc_mean,
            best_threshold_mean, n_features_mean, n (#seeds).
    """
    df = gs_df.copy()

    # Ensure threshold is numeric; non-convertible values become NaN
    if not np.issubdtype(df["threshold"].dtype, np.number):
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")

    # Use a helper column for argmax over accuracy
    df["acc_for_argmax"] = df["accuracy"].fillna(-np.inf)

    # For each (model, seed), keep the single row with max accuracy
    best_rows = (
        df.sort_values(["model", "seed", "acc_for_argmax"], ascending=[True, True, False])
        .groupby(["model", "seed"], as_index=False)
        .head(1)
        .drop(columns=["acc_for_argmax"])
    )

    # Aggregate those best-per-seed rows across seeds
    best_means = (
        best_rows.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            auc_mean=("auc", "mean"),
            best_threshold_mean=("threshold", "mean"),
            n_features_mean=("n_selected", "mean"),
            n=("accuracy", "count"),
        )
        .reset_index()
    )
    return best_means


def compute_best_overall_by_model(gs_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize best-per-seed SHAP runs with mean and std across seeds.

    For each (model, seed), this function selects the row with the highest accuracy
    (i.e., the best SHAP threshold for that seed). Then, for each model, it
    computes:

      - the mean threshold used (threshold_used)
      - the mean number of selected features (n_features)
      - the mean and standard deviation of accuracy, F1, precision, recall, AUROC
        across seeds
      - the number of seeds n

    This summary is used for the top-level table in the PDF report
    (`best_overall_by_model.csv`).

    Args:
        gs_df: Grid-search DataFrame across models, thresholds and seeds.

    Returns:
        DataFrame with one row per model and the following columns:
            model, seed (NaN placeholder),
            threshold_used, n_features,
            accuracy_mean, accuracy_std,
            f1_mean, f1_std,
            precision_mean, precision_std,
            recall_mean, recall_std,
            auc_mean, auc_std,
            n
    """
    df = gs_df.copy()

    # Ensure threshold is numeric
    if not np.issubdtype(df["threshold"].dtype, np.number):
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")

    # Helper column to pick argmax accuracy
    df["acc_for_argmax"] = df["accuracy"].fillna(-np.inf)

    # Best row per (model, seed) according to accuracy
    best_rows = (
        df.sort_values(["model", "seed", "acc_for_argmax"], ascending=[True, True, False])
        .groupby(["model", "seed"], as_index=False)
        .head(1)
        .drop(columns=["acc_for_argmax"])
    )

    def _std_unbiased(x: pd.Series) -> float:
        """Unbiased sample standard deviation (NaN if <= 1 observation)."""
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan

    # Aggregate over seeds to get mean and std per model
    agg = (
        best_rows.groupby("model")
        .agg(
            threshold_used=("threshold", "mean"),
            n_features=("n_selected", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", _std_unbiased),
            f1_mean=("f1", "mean"),
            f1_std=("f1", _std_unbiased),
            precision_mean=("precision", "mean"),
            precision_std=("precision", _std_unbiased),
            recall_mean=("recall", "mean"),
            recall_std=("recall", _std_unbiased),
            auc_mean=("auc", "mean"),
            auc_std=("auc", _std_unbiased),
            n=("accuracy", "count"),
        )
        .reset_index()
    )

    # Seed column is kept for layout consistency in the table (left empty)
    agg["seed"] = np.nan

    cols = [
        "model",
        "seed",
        "threshold_used",
        "n_features",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "auc_mean",
        "auc_std",
        "n",
    ]
    agg = agg[cols]
    return agg.sort_values("model").reset_index(drop=True)


def per_model_threshold_curves(gs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean metric curves per model × threshold across seeds.

    This is used to plot:
      - metric trends vs threshold (accuracy, F1, precision, recall)
      - number of selected features vs threshold

    Args:
        gs_df: Grid-search DataFrame across models, thresholds and seeds.

    Returns:
        DataFrame with columns:
            model, threshold, accuracy, f1, precision, recall, auc, n_selected
        where values are means across seeds for each (model, threshold).
    """
    df = gs_df.copy()

    # Ensure threshold is numeric; this is crucial for sorting and plotting
    if not np.issubdtype(df["threshold"].dtype, np.number):
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")

    curves = (
        df.groupby(["model", "threshold"])
        .agg(
            accuracy=("accuracy", "mean"),
            f1=("f1", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            auc=("auc", "mean"),
            n_selected=("n_selected", "mean"),
        )
        .reset_index()
    )
    return curves


# ----------------------------
# Plotting primitives
# ----------------------------

def _add_text_page(
    pdf: PdfPages,
    title: str,
    text: str,
    figsize: tuple[float, float] = (11, 3.0),
) -> None:
    """Add a purely textual page to the PDF.

    This is used to introduce each section or figure with an explanatory caption.

    Args:
        pdf: Open PdfPages object to append the page to.
        title: Page title shown at the top-left.
        text: Body text shown below the title.
        figsize: Figure size in inches (width, height).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, pad=12)
    fig.text(0.02, 0.80, text, ha="left", va="top", fontsize=10, wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _bar_baseline_vs_best(
    pdf: PdfPages,
    model: str,
    baseline_row: pd.Series,
    best_row: pd.Series,
) -> None:
    """Plot a bar chart: baseline vs SHAP best-per-seed metrics for one model.

    The chart compares:
      - baseline: model trained on all features, metrics averaged across seeds
      - SHAP best-per-seed: model retrained on SHAP-selected features, taking
        for each seed the threshold that maximizes accuracy and averaging metrics
        across seeds.

    Args:
        pdf: Open PdfPages object.
        model: Model name (used in title).
        baseline_row: Series with baseline mean metrics (accuracy, f1, precision,
            recall, auc).
        best_row: Series with SHAP best-per-seed mean metrics, with keys
            "accuracy_mean", "f1_mean", etc.
    """
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]
    base_vals = [float(baseline_row.get(m, np.nan)) for m in metrics]
    shap_vals = [float(best_row.get(f"{m}_mean", np.nan)) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.bar(x - width / 2, base_vals, width, label="baseline (mean over seeds)")
    ax.bar(x + width / 2, shap_vals, width, label="SHAP best-per-seed (mean)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(f"{model} — baseline vs SHAP best-per-seed (incl. AUROC)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _lines_metrics_vs_threshold(pdf: PdfPages, model: str, sub: pd.DataFrame) -> None:
    """Plot metric trends (accuracy, F1, precision, recall) vs threshold.

    Args:
        pdf: Open PdfPages object.
        model: Model name (used in title).
        sub: Subset of the curves DataFrame for a single model.
    """
    metrics = ["accuracy", "f1", "precision", "recall"]
    thr = sorted(sub["threshold"].dropna().unique())
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    # For each metric, extract the mean value at each threshold
    for m in metrics:
        y = [
            float(sub.loc[sub["threshold"] == t, m].values[0])
            if not sub.loc[sub["threshold"] == t, m].empty
            else np.nan
            for t in thr
        ]
        ax.plot(thr, y, marker="o", label=m)

    ax.set_title(f"{model} — metric trends vs threshold (mean over seeds)")
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _lines_nselected_vs_threshold(pdf: PdfPages, model: str, sub: pd.DataFrame) -> None:
    """Plot the mean number of selected features vs threshold for one model.

    Args:
        pdf: Open PdfPages object.
        model: Model name (used in title).
        sub: Subset of the curves DataFrame for a single model.
    """
    thr = sorted(sub["threshold"].dropna().unique())
    y = [
        float(sub.loc[sub["threshold"] == t, "n_selected"].values[0])
        if not sub.loc[sub["threshold"] == t, "n_selected"].empty
        else np.nan
        for t in thr
    ]

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(thr, y, marker="o")
    ax.set_title(f"{model} — number of selected features vs threshold (mean over seeds)")
    ax.set_xlabel("threshold")
    ax.set_ylabel("# selected features")
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_table_best_overall(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    max_rows_per_page: int = 24,
) -> None:
    """Add a paginated table of best-per-seed summary statistics to the PDF.

    The table shows, for each model:
      - the mean threshold and mean #features used
      - mean and standard deviation of all metrics (accuracy, F1, precision,
        recall, AUROC)
      - the number of seeds contributing to the summary (n)

    Args:
        pdf: Open PdfPages object.
        df: DataFrame produced by `compute_best_overall_by_model`.
        title: Title used for the table (and page title).
        max_rows_per_page: Maximum number of table rows per PDF page.
    """
    cols = [
        "model",
        "seed",
        "threshold_used",
        "n_features",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "auc_mean",
        "auc_std",
        "n",
    ]
    df = df[cols].copy()

    # Split into chunks to avoid overly long pages
    chunks = [df.iloc[i : i + max_rows_per_page] for i in range(0, len(df), max_rows_per_page)]
    for i, chunk in enumerate(chunks):
        fig_height = 0.40 * len(chunk) + 2.0
        fig, ax = plt.subplots(figsize=(11.5, fig_height))
        ax.axis("off")
        ax.set_title(f"{title} (page {i+1})", loc="left", fontsize=12, pad=10)

        headers = [
            "model",
            "seed",
            "thr_used",
            "#feat",
            "acc (mean)",
            "acc std",
            "F1 (mean)",
            "F1 std",
            "prec (mean)",
            "prec std",
            "rec (mean)",
            "rec std",
            "AUROC (mean)",
            "AUROC std",
            "n",
        ]
        table_data = [headers]

        # Convert each row into a list of formatted strings
        for _, r in chunk.iterrows():
            row = [
                r["model"],
                int(r["seed"]) if pd.notna(r["seed"]) else "",
                f"{float(r['threshold_used']):.4g}" if pd.notna(r["threshold_used"]) else "",
                int(r["n_features"]) if pd.notna(r["n_features"]) else "",
                *[
                    f"{float(r[c]):.4f}" if pd.notna(r[c]) else ""
                    for c in [
                        "accuracy_mean",
                        "accuracy_std",
                        "f1_mean",
                        "f1_std",
                        "precision_mean",
                        "precision_std",
                        "recall_mean",
                        "recall_std",
                        "auc_mean",
                        "auc_std",
                    ]
                ],
                int(r["n"]) if pd.notna(r["n"]) else "",
            ]
            table_data.append(row)

        table = ax.table(cellText=table_data, loc="center", cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ----------------------------
# Main builder
# ----------------------------

def build_threshold_aggregate_report(base_folder: str) -> Tuple[str, str, str]:
    """High-level pipeline to build the threshold-focused aggregate report.

    Steps:
      1. Load per-seed metrics (`metrics_report.csv`) and grid-search results
         (`gridsearch_thresholds.csv`) from `base_folder`.
      2. Compute:
           - baseline_means: mean baseline metrics per model
           - best_per_seed: mean metrics at best-per-seed threshold per model
           - best_overall: mean ± std of best-per-seed metrics per model
           - curves: mean metric curves per model × threshold
      3. Restrict all summaries to models present in all four tables.
      4. Save:
           - best_per_seed_means.csv
           - best_overall_by_model.csv
      5. Generate:
           - gridsearch_threshold_report.pdf

    Args:
        base_folder: Base directory containing the `report/seed_*` subfolders.

    Returns:
        Tuple with:
          (pdf_path, best_per_seed_means_csv_path, best_overall_csv_path)
    """
    out_dir = os.path.join(base_folder, "report_threshold")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load per-seed metrics and grid-search results
    df_metrics = load_all_seed_metrics(base_folder)
    gs_df = load_all_gridsearch(base_folder)

    # 2) Compute summary tables
    baseline_means = compute_baseline_means(df_metrics)
    best_per_seed = compute_best_per_seed_means(gs_df)
    best_overall = compute_best_overall_by_model(gs_df)
    curves = per_model_threshold_curves(gs_df)

    # Consider only models that appear in all summaries
    common_models = sorted(
        set(baseline_means["model"])
        & set(best_per_seed["model"])
        & set(best_overall["model"])
        & set(curves["model"])
    )
    baseline_means = baseline_means[baseline_means["model"].isin(common_models)].reset_index(drop=True)
    best_per_seed = best_per_seed[best_per_seed["model"].isin(common_models)].reset_index(drop=True)
    best_overall = best_overall[best_overall["model"].isin(common_models)].reset_index(drop=True)
    curves = curves[curves["model"].isin(common_models)].reset_index(drop=True)

    # 3) Save CSV summaries
    best_per_seed_csv = os.path.join(out_dir, "best_per_seed_means.csv")
    best_per_seed.to_csv(best_per_seed_csv, index=False)

    best_overall_csv = os.path.join(out_dir, "best_overall_by_model.csv")
    best_overall.to_csv(best_overall_csv, index=False)

    # 4) Build PDF report
    pdf_path = os.path.join(out_dir, "gridsearch_threshold_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Summary page + table of best overall stats
        _add_text_page(
            pdf,
            title="Summary — Best overall per model (mean ± std over per-seed bests)",
            text=(
                "For each model, within each seed we select the SHAP-Select threshold "
                "maximizing accuracy (best-per-seed). We then report MEAN and STANDARD "
                "DEVIATION across seeds for all metrics at those points. threshold_used "
                "and n_features are the corresponding means across seeds."
            ),
        )
        _add_table_best_overall(pdf, best_overall, title="Best overall per model (mean ± std across seeds)")

        # Per-model sections: histogram + curves
        for mdl in common_models:
            base_row = baseline_means[baseline_means["model"] == mdl]
            best_row = best_per_seed[best_per_seed["model"] == mdl]
            curve_row = curves[curves["model"] == mdl].sort_values("threshold")

            # Skip models with incomplete data
            if base_row.empty or best_row.empty or curve_row.empty:
                continue

            base_row = base_row.iloc[0]
            best_row = best_row.iloc[0]

            # Histogram: baseline vs SHAP best-per-seed
            _add_text_page(
                pdf,
                title=f"{mdl} — Baseline vs SHAP best-per-seed (histogram)",
                text=(
                    "Comparison between the baseline model (all features) and the SHAP-Select model obtained "
                    "by picking, in each seed, the threshold maximizing accuracy and averaging metrics over seeds. "
                    "Bars show mean scores for accuracy, F1, precision, recall, and AUROC."
                ),
            )
            _bar_baseline_vs_best(pdf, mdl, base_row, best_row)

            # Metric trends vs threshold
            _add_text_page(
                pdf,
                title=f"{mdl} — Metrics vs threshold",
                text=(
                    "Mean performance across seeds as the SHAP-Select threshold changes. "
                    "Each line corresponds to accuracy, F1, precision, or recall."
                ),
            )
            _lines_metrics_vs_threshold(pdf, mdl, curve_row)

            # Number of selected features vs threshold
            _add_text_page(
                pdf,
                title=f"{mdl} — #Features selected vs threshold",
                text=(
                    "Average number of features selected by SHAP-Select at each threshold across seeds."
                ),
            )
            _lines_nselected_vs_threshold(pdf, mdl, curve_row)

    print(f"[THRESHOLD-AGG] PDF saved: {pdf_path}")
    print(f"[THRESHOLD-AGG] Best-per-seed means CSV saved: {best_per_seed_csv}")
    print(f"[THRESHOLD-AGG] Best-overall-by-model CSV saved: {best_overall_csv}")
    return pdf_path, best_per_seed_csv, best_overall_csv


def main() -> None:
    """CLI entry point.

    Example:
        python build_threshold_report.py --base results/fs_shap
    """
    parser = argparse.ArgumentParser(
        description="Build threshold-focused aggregate report from existing CSVs."
    )
    parser.add_argument(
        "--base",
        default="results/fs_shap",
        help="Base folder that contains report/seed_* subfolders.",
    )
    args = parser.parse_args()
    build_threshold_aggregate_report(args.base)


if __name__ == "__main__":
    main()
