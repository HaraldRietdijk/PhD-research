
from typing import Optional
import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# aggregate function to compute 95% CI
def _ci95(series: pd.Series) -> float:
    """95% CI half-width with normal approximation."""
    x = series.dropna().astype(float)
    n = len(x)
    if n <= 1:
        return np.nan
    return 1.96 * x.std(ddof=1) / math.sqrt(n)


def _load_all_seed_metrics(base_folder: str) -> pd.DataFrame:
    """Load all per-seed metrics_report.csv files."""
    paths = sorted(glob.glob(os.path.join(base_folder, "report", "seed_*", "metrics_report.csv")))
    frames = []
    for p in paths:
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


def _plot_aggregate_bars(df_metrics: pd.DataFrame, out_dir: str):
    """
    Bar charts (base vs shap) using mean ± 95% CI per model for each metric.
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]

    # aggregate per model × variant
    agg = (df_metrics
           .groupby(["model", "variant"])
           .agg(accuracy_mean=("accuracy","mean"), accuracy_ci95=("accuracy", _ci95),
                f1_mean=("f1","mean"), f1_ci95=("f1", _ci95),
                precision_mean=("precision","mean"), precision_ci95=("precision", _ci95),
                recall_mean=("recall","mean"), recall_ci95=("recall", _ci95),
                auc_mean=("auc","mean"), auc_ci95=("auc", _ci95))
           .reset_index())

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
        ax.bar(x - width/2, base_mean, width, yerr=base_ci, capsize=4, label="base")
        ax.bar(x + width/2, shap_mean, width, yerr=shap_ci, capsize=4, label="shap")
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


def _write_feature_summary_tables(all_feature_records: list, out_dir: str) -> pd.DataFrame:
    """
    Build feature selection frequency and mean p-value (only when selected) per model×feature.
    Saves CSV and returns summary DataFrame.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not all_feature_records:
        raise RuntimeError("No feature records collected. Make sure you append them during the run.")

    df = pd.DataFrame(all_feature_records)  # cols: seed, model, feature, p_value
    # frequency = how many seeds selected it
    freq = df.groupby(["model", "feature"]).size().rename("times_selected").reset_index()
    mean_p = df.groupby(["model", "feature"])["p_value"].mean().rename("mean_p_value").reset_index()
    summary = (freq.merge(mean_p, on=["model", "feature"])
                    .sort_values(["model", "times_selected", "mean_p_value"], ascending=[True, False, True]))
    out_csv = os.path.join(out_dir, "feature_selection_summary.csv")
    summary.to_csv(out_csv, index=False)
    return summary


def _add_feature_tables_to_pdf(summary: pd.DataFrame, pdf: PdfPages, max_rows_per_page: int = 30):
    """
    Add per-model tables (feature, times_selected, mean_p_value) into the PDF.
    Splits over multiple pages if needed.
    """
    for model in summary["model"].unique():
        sub = summary[summary["model"] == model].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["times_selected", "mean_p_value"], ascending=[False, True])

        # Chunk rows to multiple pages if necessary
        chunks = [sub.iloc[i:i+max_rows_per_page] for i in range(0, len(sub), max_rows_per_page)]
        for idx, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(10, 0.35*len(chunk) + 1.5))
            ax.axis("off")
            ax.set_title(f"{model} — Selected features (freq & mean p-value)", loc="left", fontsize=12, pad=10)

            table_data = [["feature", "times_selected", "mean_p_value"]]
            for _, r in chunk.iterrows():
                table_data.append([r["feature"], int(r["times_selected"]), f"{r['mean_p_value']:.4g}"])

            table = ax.table(cellText=table_data, loc="center", cellLoc="left", colLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _build_aggregate_pdf(base_folder: str, df_metrics: pd.DataFrame, feature_summary: pd.DataFrame):
    """
    Create a dedicated PDF with:
     - aggregate metric bars (base vs shap, mean ± 95% CI)
     - per-model feature selection tables (times_selected, mean_p_value)
    """
    out_dir = os.path.join(base_folder, "report_aggregate")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "aggregate_report.pdf")

    with PdfPages(pdf_path) as pdf:
        # 1) Aggregate bars (one page per metric)
        metrics = ["accuracy", "f1", "precision", "recall", "auc"]
        agg = (df_metrics
               .groupby(["model", "variant"])
               .agg(accuracy_mean=("accuracy","mean"), accuracy_ci95=("accuracy", _ci95),
                    f1_mean=("f1","mean"), f1_ci95=("f1", _ci95),
                    precision_mean=("precision","mean"), precision_ci95=("precision", _ci95),
                    recall_mean=("recall","mean"), recall_ci95=("recall", _ci95),
                    auc_mean=("auc","mean"), auc_ci95=("auc", _ci95))
               .reset_index())

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
            ax.bar(x - width/2, base_mean, width, yerr=base_ci, capsize=4, label="base")
            ax.bar(x + width/2, shap_mean, width, yerr=shap_ci, capsize=4, label="shap")
            ax.set_title(f"{m}: mean ± 95% CI (across seeds)")
            ax.set_ylabel(m)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 2) Feature selection tables (split across pages)
        _add_feature_tables_to_pdf(feature_summary, pdf)

    print(f"[AGG] PDF saved: {pdf_path}")
    return pdf_path