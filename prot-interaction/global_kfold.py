"""Experiment 2: one model for the whole dataset, k-fold cross-validation.

Rows from every species are pooled and split into stratified folds, so each
test fold contains species the model has also seen during training.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import Config
from ..data import FeatureStore
from ..metrics import classification_metrics, fmt, pct_ones, per_species_breakdown
from ..plots import save_confusion_matrix, save_per_species_bar, save_roc_curve
from ..training import train_and_evaluate


def run_global_kfold(store: FeatureStore, cfg: Config) -> dict:
    """Run the experiment, write plots + an .xlsx report, return the OOF results."""
    k = cfg.n_folds
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: single model, {k}-fold cross-validation over the entire dataset")
    print("=" * 70)

    labels, species, n_rows = store.labels, store.species, store.n_rows
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)

    fold_rows = []
    oof_probs = np.full(n_rows, np.nan)   # out-of-fold probabilities
    oof_preds = np.full(n_rows, np.nan)   # out-of-fold hard predictions

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_rows), labels), start=1):
        print(f"  running fold {fold_i}/{k} on {n_rows} rows...")
        res = train_and_evaluate(
            store.subset(train_idx), store.subset(test_idx),
            f"global_fold{fold_i}", cfg, make_plots=False, verbose=False)

        oof_probs[test_idx] = res.probs
        oof_preds[test_idx] = res.preds
        fold_rows.append({
            "fold": fold_i, "n_train": len(train_idx), "n_test": len(test_idx),
            "pct_ones_train": pct_ones(labels[train_idx]),
            "pct_ones_test": pct_ones(labels[test_idx]),
            "accuracy": res.accuracy, "auc": res.auc,
            "precision": res.precision, "recall": res.recall,
        })
        print(f"    fold {fold_i}/{k}: acc={res.accuracy:.4f}  auc={fmt(res.auc)}")

    # Metrics over all out-of-fold predictions combined.
    oof_preds = oof_preds.astype(int)
    m = classification_metrics(labels, oof_preds, oof_probs)
    cm = m["confusion_matrix"]
    print(f"  {k}-fold OOF accuracy={m['accuracy']:.4f}  auc={fmt(m['auc'])}  "
          f"precision={m['precision']:.4f}  recall={m['recall']:.4f}")

    split = f"global_{k}fold"
    save_confusion_matrix(cm, split, cfg)
    if len(np.unique(labels)) == 2:
        save_roc_curve(labels, oof_probs, split, cfg)
    species_stats = per_species_breakdown(species, labels, oof_preds)
    if len(species_stats) > 1:
        save_per_species_bar(species_stats, split, cfg)

    fold_df = pd.DataFrame(fold_rows)
    excel_path = os.path.join(cfg.output_dir, "experiment2_global_kfold.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([{
            "n_folds": k, "n_rows": n_rows, "n_species": len(pd.unique(species)),
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "seed": cfg.seed, "accuracy": m["accuracy"], "auc": m["auc"],
            "precision": m["precision"], "recall": m["recall"],
        }]).to_excel(writer, sheet_name="experiment_details", index=False)
        fold_df.to_excel(writer, sheet_name="fold_details", index=False)
        pd.DataFrame(cm, index=["actual_0", "actual_1"],
                     columns=["pred_0", "pred_1"]).to_excel(writer, sheet_name="confusion_matrix")
        species_stats.reset_index().to_excel(writer, sheet_name="per_species_breakdown",
                                             index=False)
    print(f"Experiment 2 report written to {excel_path}")

    return {**m, "species_stats": species_stats, "probs": oof_probs,
            "preds": oof_preds, "labels": labels, "fold_details": fold_df}
