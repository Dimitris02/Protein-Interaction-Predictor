"""Train and evaluate one model on one train/test split."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import RowDataset
from .metrics import classification_metrics, fmt, per_species_breakdown
from .models import build_model
from .plots import (save_confusion_matrix, save_loss_curve,
                    save_per_species_bar, save_roc_curve)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed torch and numpy for reproducible runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class EvalResult:
    """Everything an experiment needs from one trained model."""

    split: str
    accuracy: float
    auc: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray
    species_stats: pd.DataFrame
    probs: np.ndarray
    preds: np.ndarray
    labels: np.ndarray


def weighted_bce(probs: torch.Tensor, target: torch.Tensor, w0: float, w1: float) -> torch.Tensor:
    """Binary cross-entropy with per-sample class weights (negatives w0, positives w1)."""
    weight = torch.where(target > 0.5, torch.full_like(target, w1), torch.full_like(target, w0))
    return F.binary_cross_entropy(probs, target, weight=weight)


def train_and_evaluate(train_ds: RowDataset, test_ds: RowDataset, split_name: str,
                       cfg: Config, make_plots: bool = True,
                       verbose: bool = True) -> EvalResult:
    """Train a fresh model on ``train_ds`` and evaluate it on ``test_ds``.

    Args:
        make_plots: if False, skip all figures and just return the numbers
            (used for cross-validation folds, which are summarised together).
        verbose: if False, silence the per-epoch log lines.
    """
    device = get_device()
    if verbose:
        print(f"\n=== Running split: {split_name} ===")
        print(f"Train rows: {len(train_ds)}  Test rows: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    model = build_model(cfg.model_name, train_ds.store.n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Balanced class weights: w_c = N / (2 * n_c), so rare classes count more.
    train_labels = train_ds.split_labels
    n0 = int((train_labels == 0).sum())
    n1 = int((train_labels == 1).sum())
    total = len(train_labels)
    w0, w1 = total / (2.0 * max(n0, 1)), total / (2.0 * max(n1, 1))
    if verbose:
        print(f"[{split_name}] Class counts in train set -> class 0: {n0}, class 1: {n1}")

    # ---------------------------- training loop ----------------------------
    train_losses, test_losses = [], []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            probs = torch.sigmoid(model(X).squeeze(1))
            loss = weighted_bce(probs, y, w0, w1)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
        train_losses.append(running / len(train_ds))

        # The validation loss is unweighted so it reflects the raw test set.
        model.eval()
        running = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                probs = torch.sigmoid(model(X).squeeze(1))
                running += F.binary_cross_entropy(probs, y).item() * X.size(0)
        test_losses.append(running / len(test_ds))

        if verbose:
            print(f"[{split_name}] Epoch {epoch:3d}/{cfg.epochs}  "
                  f"train_loss={train_losses[-1]:.4f}  test_loss={test_losses[-1]:.4f}")

    # ----------------------- final predictions on test ---------------------
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            probs = torch.sigmoid(model(X.to(device)).squeeze(1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs >= 0.5).astype(int)

    m = classification_metrics(labels, preds, probs)
    species_stats = per_species_breakdown(test_ds.split_species, labels, preds)

    if make_plots:
        save_loss_curve(train_losses, test_losses, split_name, cfg)
        save_confusion_matrix(m["confusion_matrix"], split_name, cfg)
        if len(species_stats) > 1:
            save_per_species_bar(species_stats, split_name, cfg)
        if len(np.unique(labels)) == 2:
            save_roc_curve(labels, probs, split_name, cfg)

    if verbose:
        print(f"[{split_name}] Overall test accuracy: {m['accuracy']:.4f}  "
              f"AUC: {fmt(m['auc'])}  Precision: {m['precision']:.4f}  "
              f"Recall: {m['recall']:.4f}")

    return EvalResult(split=split_name, species_stats=species_stats,
                      probs=probs, preds=preds, labels=labels, **m)
