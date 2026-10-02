# Species Classifier

Species-level binary classifier for large feature tables. Each CSV row holds two
concatenated embeddings (e.g. two 1024-d vectors) and a 0/1 label; the model
predicts the label for the pair.

The CSV is streamed once into an on-disk `numpy.memmap`, so datasets far larger
than RAM work fine. Only integer row indices are kept in memory.

## Experiments

| # | Name | What it answers |
|---|------|-----------------|
| 1 | Per-species k-fold | How well can a model learn *within* one species? (one model per species) |
| 2 | Global k-fold | How well does one model do on the pooled dataset? |
| 3 | Leave-species-out | Does the model generalise to species it never saw in training? |

Each experiment writes plots (`.png`) and an Excel report (`.xlsx`) to the output directory.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Install `torch` following [pytorch.org](https://pytorch.org/get-started/locally/) if you need a specific CUDA build.)

## Data format

`data.csv` with one row per data point:

| Column | Description |
|--------|-------------|
| `filename` | species name (e.g. `Homo_Sapiens`) |
| `combined_score` | label, `0` or `1` |
| *all other columns* | numeric input features (2048 by default: `[A \| B]`) |

## Usage

```bash
# run everything with defaults
python -m species_classifier --csv data.csv

# only experiment 2, custom hyperparameters
python -m species_classifier --csv data.csv --experiments 2 --epochs 50 --lr 5e-4

# choose the held-out species for experiment 3
python -m species_classifier --csv data.csv --experiments 3 \
    --holdout-species Homo_Sapiens Mus_Musculus
```

Run `python -m species_classifier --help` for every option.

> **Cache note:** the first run builds `cache/features.mmap`. If you change
> `data.csv`, pass `--rebuild-cache` (or delete the `cache/` folder).

## Project layout

```
species_classifier/
├── cli.py                  # argument parsing + experiment orchestration
├── config.py               # Config dataclass (all hyperparameters/paths)
├── data.py                 # CSV -> memmap cache, FeatureStore, RowDataset
├── models.py               # Siamese and MLP architectures
├── training.py             # train/evaluate one split
├── metrics.py              # accuracy/AUC/per-species helpers
├── plots/
│   ├── style.py            # matplotlib theme (headless Agg backend)
│   └── figures.py          # every figure the project produces
└── experiments/
    ├── per_species_kfold.py
    ├── global_kfold.py
    └── holdout_species.py
```

## Models

- `siamese` (default): a shared encoder embeds both halves of the input; a head
  scores `[A*B, |A-B|]`.
- `mlp`: a plain MLP on `[A*B, |A-B|]`.

Select with `--model`. Training uses Adam and class-balanced binary
cross-entropy.

## Notes

- Plot titles and axis labels are in Greek; edit `plots/figures.py` to change them.
- Runs are seeded (`--seed`), but GPU kernels can still introduce tiny nondeterminism.
