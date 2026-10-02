"""The three evaluation experiments."""

from .global_kfold import run_global_kfold
from .holdout_species import run_holdout_species
from .per_species_kfold import run_per_species_kfold

__all__ = ["run_per_species_kfold", "run_global_kfold", "run_holdout_species"]
