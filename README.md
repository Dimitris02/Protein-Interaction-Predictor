# Protein–Protein Interaction Prediction

This repository stores all the scripts that were used in my [master's thesis](https://olympias.lib.uoi.gr/jspui/handle/123456789/40442)

Predicts whether two proteins **interact** or **do not interact**, using only
their amino-acid–derived embeddings.

Every protein is represented by a 1024-dimensional sequence embedding from
[SPACE](https://doi.org/10.1093/bioinformatics/btaf496) (*STRING Proteins as
Complementary Embeddings*), which is distributed with the
[STRING database](https://string-db.org/cgi/download). A training example is a
protein pair: the two embeddings are concatenated into one 2048-d row
(`[A | B]`) and the model outputs the probability that the pair interacts.

The pair features are symmetric (`A*B` and `|A-B|`), so the prediction does not
depend on the order of the two proteins.

The feature table can be far larger than RAM: the CSV is streamed once into an
on-disk `numpy.memmap` and only integer row indices are kept in memory.

The data.csv we used is available [here](https://huggingface.co/datasets/Dimitris02/Protein_interractions/tree/main) (place it in the prot-interaction folder to use it).

## Experiments

Each protein pair belongs to a species, which lets us ask three different
questions about how well the model generalises:

| # | Name | What it answers |
|---|------|-----------------|
| 1 | Per-species k-fold | How well can interactions be predicted *within* a single species? (one model per species) |
| 2 | Global k-fold | How well does one model do on all species pooled together? |
| 3 | Leave-species-out | Does the model transfer to species it never saw during training? |

Each experiment writes plots (`.png`) and an Excel report (`.xlsx`) to the output directory.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Install `torch` following [pytorch.org](https://pytorch.org/get-started/locally/) if you need a specific CUDA build.)

## Data format

A single `data.csv` with one row per protein pair:

| Column | Description |
|--------|-------------|
| `filename` | species the pair belongs to (e.g. `Homo_Sapiens`) |
| `combined_score` | label: `1` = interaction, `0` = no interaction |
| *remaining 2048 columns* | SPACE sequence embeddings of the two proteins: the first 1024 values are protein A, the last 1024 are protein B |

Note that the two halves of each row must follow this A-then-B layout, because
the models split every row in the middle.

Use the **sequence** embeddings from SPACE (derived from the amino-acid chain).
SPACE also ships *network* embeddings, which are computed from STRING's own
interaction network and would leak interaction information into the features.

## Usage

```bash
# run everything with defaults
python -m species_classifier --csv data.csv

# only experiment 2, custom hyperparameters
python -m species_classifier --csv data.csv --experiments 2 --epochs 50 --lr 5e-4

# choose the species held out in experiment 3
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

Both models take the pair `[A | B]` and output one interaction logit.

- `siamese` (default): a shared encoder embeds protein A and protein B with the
  same weights; a classification head then scores `[A'*B', |A'-B'|]`.
- `mlp`: a plain MLP applied directly to `[A*B, |A-B|]`.

Select with `--model`. Training uses Adam and class-balanced binary
cross-entropy, which compensates for interacting and non-interacting pairs
being unequally frequent.

## Notes

- Plot titles and axis labels are in Greek; edit `plots/figures.py` to change them.
- Runs are seeded (`--seed`), but GPU kernels can still introduce tiny nondeterminism.

## Citation

If you use the embeddings, please cite SPACE:

> Hu D., Szklarczyk D., von Mering C., Jensen L. J. *SPACE: STRING proteins as
> complementary embeddings.* Bioinformatics 41(8), 2025.
