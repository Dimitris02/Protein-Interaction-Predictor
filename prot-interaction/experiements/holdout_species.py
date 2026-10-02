"""Experiment 3: leave-species-out generalisation.

Train on every species EXCEPT the held-out ones, then test on exactly the
held-out species. This shows how well the model transfers to unseen species.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..data import FeatureStore
from ..metrics import pct_ones
from ..training import EvalResult, train_and_evaluate


def run_holdout_species(store: FeatureStore, cfg: Config) -> Optional[EvalResult]:
    """Run the experiment; returns None if no held-out species is in the data."""
    holdout = list(cfg.holdout_species)
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: train on all species except the held-out ones, "
          "test on the held-out species")
    print("=" * 70)
    print(f"Held-out species: {holdout}")

    species, labels = store.species, store.labels
    is_holdout = np.isin(species, holdout)
    found = set(pd.unique(species[is_holdout]))
    missing = set(holdout) - found
    if missing:
        print(f"  WARNING: these requested species were not found in the data "
              f"(check exact spelling/underscores): {sorted(missing)}")

    train_idx = np.where(~is_holdout)[0]
    test_idx = np.where(is_holdout)[0]
    if len(test_idx) == 0:
        print("  No rows found for the requested held-out species -- skipping experiment 3.")
        return None

    res = train_and_evaluate(store.subset(train_idx), store.subset(test_idx),
                             "holdout_species", cfg, make_plots=True, verbose=True)

    excel_path = os.path.join(cfg.output_dir, "experiment3_holdout_species.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([{
            "held_out_species": ", ".join(holdout),
            "n_train": len(train_idx), "n_test": len(test_idx),
            "pct_ones_train": pct_ones(labels[train_idx]),
            "pct_ones_test": pct_ones(labels[test_idx]),
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "accuracy": res.accuracy, "auc": res.auc,
            "precision": res.precision, "recall": res.recall,
        }]).to_excel(writer, sheet_name="experiment_details", index=False)
        pd.DataFrame(res.confusion_matrix, index=["actual_0", "actual_1"],
                     columns=["pred_0", "pred_1"]).to_excel(writer, sheet_name="confusion_matrix")
        res.species_stats.reset_index().to_excel(writer, sheet_name="per_species_breakdown",
                                                 index=False)
    print(f"Experiment 3 report written to {excel_path}")
    return res
