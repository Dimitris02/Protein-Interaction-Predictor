"""Experiment 1: one model per species, evaluated with k-fold cross-validation.

This measures how well a model can learn *within* a single species. Species
with too little data (or too few examples of either class) are skipped.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import Config
from ..data import FeatureStore
from ..metrics import classification_metrics, fmt, pct_ones
from ..plots import save_confusion_matrix, save_kfold_summary
from ..training import train_and_evaluate


def run_per_species_kfold(store: FeatureStore, cfg: Config):
    """Run the experiment; returns ``(fold_df, summary_df)`` and writes an .xlsx report."""
    k = cfg.n_folds
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 1: per-species models, {k}-fold cross-validation")
    print("=" * 70)

    labels, species = store.labels, store.species
    fold_rows, summary_rows = [], []

    for sp in sorted(pd.unique(species)):
        sp_idx = np.where(species == sp)[0]
        sp_labels = labels[sp_idx]
        n0 = int((sp_labels == 0).sum())
        n1 = int((sp_labels == 1).sum())

        # Stratified k-fold needs enough samples of both classes.
        if len(sp_idx) < k * 2 or n0 < cfg.min_per_class or n1 < cfg.min_per_class:
            print(f"  [{sp}] skipped -- not enough data/class diversity "
                  f"(n={len(sp_idx)}, class0={n0}, class1={n1})")
            continue

        print(f"  [{sp}] running {k}-fold CV on {len(sp_idx)} rows...")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)

        # Out-of-fold (OOF) predictions: every row is predicted exactly once,
        # by the model that never saw it during training.
        oof_probs = np.full(len(sp_idx), np.nan)
        oof_preds = np.full(len(sp_idx), np.nan)

        for fold_i, (tr_pos, te_pos) in enumerate(skf.split(sp_idx, sp_labels), start=1):
            train_idx, test_idx = sp_idx[tr_pos], sp_idx[te_pos]
            res = train_and_evaluate(
                store.subset(train_idx), store.subset(test_idx),
                f"per_species_{sp}_fold{fold_i}", cfg, make_plots=False, verbose=False)

            oof_probs[te_pos] = res.probs
            oof_preds[te_pos] = res.preds
            fold_rows.append({
                "species": sp, "fold": fold_i,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "pct_ones_train": pct_ones(labels[train_idx]),
                "pct_ones_test": pct_ones(labels[test_idx]),
                "accuracy": res.accuracy, "auc": res.auc,
                "precision": res.precision, "recall": res.recall,
            })
            print(f"    fold {fold_i}/{k}: acc={res.accuracy:.4f}  auc={fmt(res.auc)}")

        # Aggregate the OOF predictions into one set of per-species metrics.
        oof_preds = oof_preds.astype(int)
        m = classification_metrics(sp_labels, oof_preds, oof_probs)
        cm = m["confusion_matrix"]
        save_confusion_matrix(cm, f"per_species_{sp}_{k}fold", cfg)

        summary_rows.append({
            "species": sp, "n_total": len(sp_idx), "n_class0": n0, "n_class1": n1,
            "pct_ones": pct_ones(sp_labels),
            "accuracy": m["accuracy"], "auc": m["auc"],
            "precision": m["precision"], "recall": m["recall"],
            "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1],
        })
        print(f"  [{sp}] {k}-fold accuracy={m['accuracy']:.4f}  auc={fmt(m['auc'])}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    save_kfold_summary(summary_df, cfg)

    excel_path = os.path.join(cfg.output_dir, "experiment1_per_species_kfold.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([{
            "n_folds": k, "epochs": cfg.epochs, "batch_size": cfg.batch_size,
            "lr": cfg.lr, "seed": cfg.seed, "min_per_class": cfg.min_per_class,
            "n_species_run": len(summary_df), "n_species_total": len(pd.unique(species)),
        }]).to_excel(writer, sheet_name="config", index=False)
        fold_df.to_excel(writer, sheet_name="fold_details", index=False)
        summary_df.to_excel(writer, sheet_name="species_summary", index=False)
    print(f"Experiment 1 report written to {excel_path}")

    return fold_df, summary_df
