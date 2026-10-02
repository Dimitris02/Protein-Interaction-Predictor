"""Lazy, memory-safe access to the feature CSV.

The CSV can be far larger than RAM, so it is streamed once into a float32
``numpy.memmap`` on disk (the "cache"). Afterwards only integer row indices
are kept in memory; feature vectors are read from disk on demand.

Expected CSV layout (one row per data point)::

    filename        species name (string)
    combined_score  binary label (0 or 1)
    <other columns> numeric model input features (e.g. 2048 of them)
"""

import os
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPECIES_COL = "filename"
LABEL_COL = "combined_score"


@dataclass(frozen=True, eq=False)
class FeatureStore:
    """Handle to the on-disk feature matrix plus its (small) metadata."""

    feat_path: str              # path of the float32 memmap file
    shape: Tuple[int, int]      # (n_rows, n_features)
    dtype: np.dtype
    labels: np.ndarray          # int8, shape (n_rows,)
    species: np.ndarray         # str,  shape (n_rows,)

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def n_features(self) -> int:
        return self.shape[1]

    def subset(self, indices: Sequence[int]) -> "RowDataset":
        """Return a PyTorch dataset restricted to the given row indices."""
        return RowDataset(self, indices)


def build_or_load_cache(csv_path: str, cache_dir: str, chunk_size: int = 5000,
                        rebuild: bool = False) -> FeatureStore:
    """Stream ``csv_path`` into a memmap (once) and return a FeatureStore.

    The full CSV is never held in RAM: feature columns go straight into a
    float32 memmap, and only the labels/species columns are kept as arrays.
    If a cache already exists it is reused, so delete ``cache_dir`` (or pass
    ``rebuild=True``) whenever the CSV changes.
    """
    os.makedirs(cache_dir, exist_ok=True)
    feat_path = os.path.join(cache_dir, "features.mmap")
    meta_path = os.path.join(cache_dir, "meta.npz")

    if not rebuild and os.path.exists(feat_path) and os.path.exists(meta_path):
        try:
            meta = np.load(meta_path, allow_pickle=False)
            store = FeatureStore(
                feat_path=feat_path,
                shape=tuple(int(s) for s in meta["shape"]),
                dtype=np.dtype(str(meta["dtype"])),
                labels=meta["labels"],
                species=meta["species"],
            )
            print(f"Loaded cached feature memmap: shape={store.shape}")
            return store
        except ValueError:
            # Caches written by older versions stored species as pickled objects.
            print("Cache is in an outdated format - rebuilding it.")

    print(f"Streaming {csv_path} to build the feature memmap...")

    # Read only the header to discover which columns are features.
    header = pd.read_csv(csv_path, nrows=0)
    feature_cols = [c for c in header.columns if c not in (SPECIES_COL, LABEL_COL)]
    n_features = len(feature_cols)
    print(f"Detected {n_features} feature columns.")

    # ---- pass 1: cheap columns only -> row count, labels, species --------
    species_chunks, label_chunks = [], []
    for chunk in pd.read_csv(csv_path, usecols=[SPECIES_COL, LABEL_COL],
                             chunksize=chunk_size):
        species_chunks.append(chunk[SPECIES_COL].to_numpy())
        label_chunks.append(chunk[LABEL_COL].to_numpy().astype(np.int8))
    species = np.concatenate(species_chunks).astype(str)
    labels = np.concatenate(label_chunks)
    n_rows = len(labels)
    print(f"Total rows: {n_rows}")

    # ---- pass 2: stream feature columns straight into the memmap ---------
    dtype = np.dtype(np.float32)
    feat_mmap = np.memmap(feat_path, dtype=dtype, mode="w+", shape=(n_rows, n_features))
    cursor = 0
    for chunk in pd.read_csv(csv_path, usecols=feature_cols, chunksize=chunk_size):
        n = len(chunk)
        feat_mmap[cursor:cursor + n] = chunk.to_numpy(dtype=dtype)
        cursor += n
        if (cursor // chunk_size) % 20 == 0:
            print(f"  ...written {cursor}/{n_rows} rows to memmap")
    feat_mmap.flush()
    del feat_mmap

    np.savez(meta_path, shape=np.array((n_rows, n_features)), dtype=str(dtype),
             labels=labels, species=species)
    print("Cache built.")
    return FeatureStore(feat_path, (n_rows, n_features), dtype, labels, species)


class RowDataset(Dataset):
    """Reads rows lazily from the memmap; never loads the matrix whole."""

    def __init__(self, store: FeatureStore, indices: Sequence[int]):
        self.store = store
        self.indices = np.asarray(indices)
        self._mmap = None  # opened lazily, once per worker process

    def _open(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(self.store.feat_path, dtype=self.store.dtype,
                                   mode="r", shape=self.store.shape)
        return self._mmap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.indices[i]
        x = np.asarray(self._open()[row], dtype=np.float32)
        y = np.float32(self.store.labels[row])
        return torch.from_numpy(x), torch.tensor(y)

    @property
    def split_labels(self) -> np.ndarray:
        """Labels of just this split's rows."""
        return self.store.labels[self.indices]

    @property
    def split_species(self) -> np.ndarray:
        """Species of just this split's rows."""
        return self.store.species[self.indices]
