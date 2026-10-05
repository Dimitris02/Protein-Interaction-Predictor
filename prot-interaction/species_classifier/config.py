"""Central configuration for the pipeline.

Every tunable value lives in the :class:`Config` dataclass so that the rest of
the code base never relies on module-level globals. The command-line interface
(:mod:`species_classifier.cli`) builds a ``Config`` from its arguments.
"""

from dataclasses import dataclass
from typing import Tuple

# Species kept out of training entirely in experiment 3 (leave-species-out).
DEFAULT_HOLDOUT_SPECIES: Tuple[str, ...] = (
    "Aquila_Chrysaetos",
    "Drosophila_Melanogaster",
    "Pleurotus_Ostreatus",
)


@dataclass
class Config:
    # ---- paths -----------------------------------------------------------
    csv_path: str = "data.csv"      # large input file
    cache_dir: str = "cache"        # on-disk feature memmap + metadata
    output_dir: str = "results"     # plots (.png) and reports (.xlsx)

    # ---- data loading ----------------------------------------------------
    chunk_size: int = 5000          # CSV rows read per chunk while caching
    num_workers: int = 0            # DataLoader workers; raise if I/O bound

    # ---- model -----------------------------------------------------------
    model_name: str = "siamese"     # "siamese" or "mlp" (see models.py)
    # Display name shown in plot titles (Greek: "Siamese Classifier").
    model_label: str = "Σιαμέζικος Ταξινομητής"

    # ---- optimisation ----------------------------------------------------
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 42

    # ---- experiments -----------------------------------------------------
    n_folds: int = 5                # folds for experiments 1 and 2
    # A species needs at least this many samples of EACH class to be
    # cross-validated in experiment 1.
    min_per_class: int = 5
    holdout_species: Tuple[str, ...] = DEFAULT_HOLDOUT_SPECIES

    # ---- plotting --------------------------------------------------------
    savefig_dpi: int = 150
