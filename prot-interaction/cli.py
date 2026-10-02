"""Command-line entry point: ``python -m species_classifier``."""

import argparse
import os

import pandas as pd

from .config import DEFAULT_HOLDOUT_SPECIES, Config
from .data import build_or_load_cache
from .experiments import (run_global_kfold, run_holdout_species,
                          run_per_species_kfold)
from .metrics import pct_ones
from .models import MODELS
from .plots import apply_style, save_species_distribution_donut
from .training import get_device, set_seed


def parse_args(argv=None) -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(
        prog="species_classifier",
        description="Predict protein-protein interactions from SPACE sequence embeddings.")
    p.add_argument("--csv", default=d.csv_path, help="input CSV (default: %(default)s)")
    p.add_argument("--cache-dir", default=d.cache_dir, help="memmap cache directory")
    p.add_argument("--output-dir", default=d.output_dir, help="plots/reports directory")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="ignore any existing cache and re-read the CSV")
    p.add_argument("--experiments", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3],
                   help="which experiments to run (default: all)")
    p.add_argument("--model", choices=sorted(MODELS), default=d.model_name)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--n-folds", type=int, default=d.n_folds)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--chunk-size", type=int, default=d.chunk_size)
    p.add_argument("--holdout-species", nargs="+", default=list(DEFAULT_HOLDOUT_SPECIES),
                   help="species held out in experiment 3")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config(
        csv_path=args.csv, cache_dir=args.cache_dir, output_dir=args.output_dir,
        chunk_size=args.chunk_size, num_workers=args.num_workers,
        model_name=args.model, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, n_folds=args.n_folds,
        min_per_class=args.n_folds, holdout_species=tuple(args.holdout_species),
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    apply_style()
    set_seed(cfg.seed)
    print(f"Device: {get_device()}")

    store = build_or_load_cache(cfg.csv_path, cfg.cache_dir, cfg.chunk_size,
                                rebuild=args.rebuild_cache)
    print(f"Loaded dataset: {store.n_rows} rows, {store.n_features} features, "
          f"{len(pd.unique(store.species))} species, "
          f"{pct_ones(store.labels):.2f}% positive overall")

    save_species_distribution_donut(store.species, cfg)
    print(f"Species distribution donut chart saved to {cfg.output_dir}/species_distribution_donut.png")

    if 1 in args.experiments:
        run_per_species_kfold(store, cfg)
    if 2 in args.experiments:
        run_global_kfold(store, cfg)
    if 3 in args.experiments:
        run_holdout_species(store, cfg)

    print(f"\nAll experiments finished. See '{cfg.output_dir}' for plots and .xlsx reports.")
