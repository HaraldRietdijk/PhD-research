from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os
import csv
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class MetricsEntry:
    model: str
    variant: str  # "base" | "shap"
    n_features: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc: float | None


@dataclass
class MetricsReporter:
    out_dir: str
    entries: List[MetricsEntry] = field(default_factory=list)

    def log(self, model: str, variant: str, n_features: int, metrics: Dict):
        self.entries.append(
            MetricsEntry(
                model=model,
                variant=variant,
                n_features=int(n_features),
                accuracy=float(metrics["accuracy"]),
                f1=float(metrics["f1"]),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                auc=(None if metrics.get("auc") is None else float(metrics["auc"])),
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self.entries:
            rows.append({
                "model": e.model,
                "variant": e.variant,
                "n_features": e.n_features,
                "accuracy": e.accuracy,
                "f1": e.f1,
                "precision": e.precision,
                "recall": e.recall,
                "auc": (np.nan if e.auc is None else e.auc),
            })
        return pd.DataFrame(rows)

    def save_csv(self, filename: str = "metrics_report.csv"):
        os.makedirs(self.out_dir, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(os.path.join(self.out_dir, filename), index=False)

    # Graphs
    def plot_trend_by_features(self, filename: str = "trend_by_n_features.png", title: str = "Metrics vs Nr. of features"):
        """
        Line plot: metrics (accuracy, f1, precision, recall) vs n_features (x-axis).
        """
        os.makedirs(self.out_dir, exist_ok=True)
        df = self.to_dataframe().sort_values("n_features")
        if df.empty:
            return

        x = df["n_features"].values
        plt.figure(figsize=(7, 5))
        plt.plot(x, df["accuracy"].values, label="accuracy")
        plt.plot(x, df["f1"].values,        label="f1-score")
        plt.plot(x, df["precision"].values, label="precision")
        plt.plot(x, df["recall"].values,    label="recall")
        plt.title(title)
        plt.xlabel("Nr. of Features used")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, filename), dpi=160)
        plt.close()

    def plot_baseline_vs_shap_bars(self, filename: str = "baseline_vs_shap_per_model.png", title: str = "Baseline vs SHAP-selected"):
        """
        For each model: grouped bars (base vs shap) for test metrics (accuracy, f1, precision, recall, auc).
        If data for either variant is missing, skip that model.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        df = self.to_dataframe()
        if df.empty:
            return

        # keep only the last entry per model×variant (in case of multiple logged)
        piv = df.groupby(["model","variant"]).tail(1).set_index(["model","variant"])

        models = sorted({m for m,_ in piv.index})
        metrics = ["accuracy","f1","precision","recall","auc"]

        # 1 row per metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.5*len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        x = np.arange(len(models))
        width = 0.35

        for i, met in enumerate(metrics):
            ax = axes[i]
            base_vals = []
            shap_vals = []
            kept_models = []
            for m in models:
                has_base = ("base" in piv.loc[m].index) if m in piv.index.get_level_values(0) else False
                has_shap = ("shap" in piv.loc[m].index) if m in piv.index.get_level_values(0) else False
                if not (has_base and has_shap):
                    continue
                b = piv.loc[(m,"base"), met]
                s = piv.loc[(m,"shap"), met]
                base_vals.append(np.nan if pd.isna(b) else b)
                shap_vals.append(np.nan if pd.isna(s) else s)
                kept_models.append(m)

            bx = np.arange(len(kept_models))
            ax.bar(bx - width/2, base_vals, width, label="base")
            ax.bar(bx + width/2, shap_vals, width, label="shap")
            ax.set_ylabel(met)
            ax.set_title(f"{met}")
            ax.set_xticks(bx)
            ax.set_xticklabels(kept_models, rotation=45, ha="right")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            if i == 0:
                ax.legend()

        plt.suptitle(title)
        plt.tight_layout(rect=[0,0,1,0.97])
        plt.savefig(os.path.join(self.out_dir, filename), dpi=160)
        plt.close()

    def build_report(self):
        """save CSV + both plots"""
        self.save_csv()
        self.plot_trend_by_features()
        self.plot_baseline_vs_shap_bars()
        
        

