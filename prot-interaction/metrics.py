"""Small metric helpers shared by training and the experiments."""

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_auc_score)


def pct_ones(labels) -> float:
    """Percentage of positive (1) labels; NaN for an empty array."""
    labels = np.asarray(labels)
    return 100.0 * (labels == 1).mean() if len(labels) else float("nan")


def fmt(value) -> str:
    """Format a metric for logging, printing 'n/a' for NaN/None."""
    return "n/a" if value is None or np.isnan(value) else f"{value:.4f}"


def classification_metrics(labels, preds, probs) -> dict:
    """Accuracy, precision, recall, ROC-AUC and the 2x2 confusion matrix.

    AUC is NaN when only one class is present (it is undefined then).
    """
    labels, preds = np.asarray(labels), np.asarray(preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float((preds == labels).mean()),
        "auc": auc,
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }


def per_species_breakdown(species, labels, preds) -> pd.DataFrame:
    """Per-species sample count, correct count and % correct (largest first)."""
    df = pd.DataFrame({"species": species, "label": labels, "pred": preds})
    df["correct"] = (df["label"] == df["pred"]).astype(int)
    stats = df.groupby("species").agg(total=("label", "count"), correct=("correct", "sum"))
    stats["pct_correct"] = 100 * stats["correct"] / stats["total"]
    return stats.sort_values("total", ascending=False)
