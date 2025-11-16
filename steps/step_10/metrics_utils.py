"""
metrics_utils.py

Utility functions for evaluating classification models and printing
train/test performance blocks. These helpers standardize metric
computation across all classifiers used in the SHAP-Select experiments.

The module provides:
    • safe extraction of decision scores / probabilities
    • computation of standard classification metrics (accuracy, precision,
      recall, F1, AUROC)
    • a unified console printout for train/test metrics
    • a high-level wrapper used throughout the pipeline

All functions here operate on already-trained sklearn-compatible models.
"""

import numpy as np
import math
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _decision_scores_or_proba(model, X):
    """Return a continuous decision score for metric computation.

    Models differ in how they expose continuous outputs:
        • probabilistic classifiers → `predict_proba(X)[:, 1]`
        • linear models, SVMs → `decision_function(X)`
        • fallback → `predict(X)` (discrete labels, only if nothing else exists)

    Continuous scores are needed for AUROC. If neither probabilities nor
    decision_function are available, using predictions degrades AUROC but
    at least ensures the code does not fail.

    Args:
        model: Trained sklearn-like model.
        X: Feature matrix.

    Returns:
        A 1D array of scores or probabilities for the positive class.
    """
    # Preferred: probability for the positive class (binary case)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Typical binary case: shape = (n_samples, 2)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        # Fallback: return whatever shape predict_proba produced
        return proba

    # Second best: signed distance to the decision boundary
    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    # Final fallback: predicted labels (AUROC will be poor or impossible)
    return model.predict(X)


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------

def compute_metrics(model, X, y, average: str = "binary") -> dict:
    """Compute a standard set of classification metrics.

    Metrics returned:
        • accuracy
        • precision
        • recall
        • F1
        • AUROC (if continuous scores are available)
        • confusion matrix
        • predictions and continuous scores (for further inspection)

    Args:
        model: A trained sklearn classifier.
        X: Feature matrix.
        y: Ground-truth labels.
        average: Averaging strategy (default: "binary").

    Returns:
        dict containing all computed metrics.
    """
    y_pred = model.predict(X)
    scores = _decision_scores_or_proba(model, X)

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average=average, zero_division=0)
    rec  = recall_score(y, y_pred, average=average, zero_division=0)
    f1   = f1_score(y, y_pred, average=average, zero_division=0)

    # AUROC may fail—e.g., if predictions are all of one class.
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
    """Format AUROC for printing (4 decimals or 'n/a')."""
    return f"{x:.4f}" if (x is not None) else "n/a"


# ----------------------------------------------------------------------
# Printing utilities
# ----------------------------------------------------------------------

def print_metrics_block(model_name: str, mt_train: dict, mt_test: dict) -> None:
    """Pretty console printout of metrics for both train and test sets.

    Args:
        model_name: Name of the classifier.
        mt_train: Metrics dictionary for the training set.
        mt_test: Metrics dictionary for the test set.
    """
    print(f"\n[METRICS] {model_name}")

    print(
        "  Train:",
        f"Acc={mt_train['accuracy']:.4f}",
        f"Prec={mt_train['precision']:.4f}",
        f"Rec={mt_train['recall']:.4f}",
        f"F1={mt_train['f1']:.4f}",
        f"AUC={_fmt_auc(mt_train['auc'])}",
    )
    print("   CM Train:\n", mt_train["cm"])

    print(
        "  Test :",
        f"Acc={mt_test['accuracy']:.4f}",
        f"Prec={mt_test['precision']:.4f}",
        f"Rec={mt_test['recall']:.4f}",
        f"F1={mt_test['f1']:.4f}",
        f"AUC={_fmt_auc(mt_test['auc'])}",
    )
    print("   CM Test:\n", mt_test["cm"])


# ----------------------------------------------------------------------
# High-level wrapper
# ----------------------------------------------------------------------

def evaluate_and_report(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    average: str = "binary",
):
    """Compute metrics for train/test splits and print them in a unified block.

    This wrapper:
        1. calls `compute_metrics` for train and test
        2. prints the two blocks using `print_metrics_block`
        3. returns the metric dictionaries

    It is used throughout the pipeline wherever baseline or SHAP-selected
    classifiers are evaluated.

    Args:
        model_name: Name of the classifier (used in print block).
        model: Trained sklearn classifier to evaluate.
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
        average: Metric averaging mode (default: "binary").

    Returns:
        (mt_train, mt_test): A tuple of two metric dictionaries.
    """
    mt_train = compute_metrics(model, X_train, y_train, average=average)
    mt_test  = compute_metrics(model, X_test,  y_test,  average=average)
    print_metrics_block(model_name, mt_train, mt_test)
    return mt_train, mt_test
