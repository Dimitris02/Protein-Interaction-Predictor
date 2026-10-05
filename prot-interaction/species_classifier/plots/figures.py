"""Figure-producing functions (one PNG per call, saved under ``cfg.output_dir``).

Axis labels and titles are intentionally in Greek; ``cfg.model_label`` is
interpolated into every title.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from ..config import Config
from .style import PALETTE, add_subtitle, plt, style_axes


def _save(fig, cfg: Config, filename: str) -> None:
    """Write ``fig`` to the output directory and release its memory."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    fig.savefig(os.path.join(cfg.output_dir, filename), dpi=cfg.savefig_dpi,
                bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(train_losses, test_losses, split_name: str, cfg: Config) -> None:
    """Train vs. validation BCE loss per epoch."""
    epochs = len(train_losses)
    xs = range(1, epochs + 1)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(xs, train_losses, label="Απώλεια εκπαίδευσης", color=PALETTE["primary"],
            linewidth=2.4, solid_capstyle="round")
    ax.fill_between(xs, train_losses, color=PALETTE["primary"], alpha=0.10)
    ax.plot(xs, test_losses, label="Απώλεια επικύρωσης", color=PALETTE["danger"],
            linewidth=2.4, linestyle="--", solid_capstyle="round")
    ax.fill_between(xs, test_losses, color=PALETTE["danger"], alpha=0.08)
    ax.set_xlabel("Εποχή")
    ax.set_ylabel("Απώλεια BCE")
    ax.set_title(f"Καμπύλη Απώλειας — {cfg.model_label}", loc="left")
    add_subtitle(ax, f"διαχωρισμός: {split_name}", y_pad=1.01)
    ax.set_xlim(1, max(epochs, 2))
    ax.legend(loc="upper right")
    style_axes(ax, hide_grid_x=True)
    fig.tight_layout()
    _save(fig, cfg, f"loss_curve_{split_name}.png")


def save_confusion_matrix(cm_counts: np.ndarray, split_name: str, cfg: Config) -> None:
    """2x2 confusion matrix, coloured by row-normalised percentage."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_percent = np.nan_to_num(
            cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True) * 100)

    fig, ax = plt.subplots(figsize=(7.8, 7))
    # "mako" is registered by seaborn; fall back to Blues if it is unavailable.
    cmap = "mako" if "mako" in plt.colormaps() else "Blues"
    im = ax.imshow(cm_percent, cmap=cmap, vmin=0, vmax=100)
    ax.set_title(f"Πίνακας Σύγχυσης — {cfg.model_label}", pad=28, loc="left")
    add_subtitle(ax, f"διαχωρισμός: {split_name}", y_pad=1.03)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ποσοστό (%)", fontsize=11)
    cbar.outline.set_visible(False)

    tick_labels = ["Αρνητικό", "Θετικό"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_yticks([0, 1]); ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Πρόβλεψη"); ax.set_ylabel("Πραγματικό")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate each cell with its count and percentage.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_counts[i, j]}\n({cm_percent[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if cm_percent[i, j] > 50 else PALETTE["dark"])

    fig.tight_layout()
    _save(fig, cfg, f"confusion_matrix_{split_name}.png")


def save_per_species_bar(species_stats: pd.DataFrame, split_name: str, cfg: Config) -> None:
    """Bars of total vs. correct samples per species, with a %-correct line."""
    x = np.arange(len(species_stats))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(max(10, len(species_stats) * 0.8), 8))
    ax1.bar(x - width / 2, species_stats["total"], width, label="Σύνολο δειγμάτων",
            color=PALETTE["primary"], edgecolor="white", linewidth=0.6, zorder=3)
    ax1.bar(x + width / 2, species_stats["correct"], width, label="Σωστές προβλέψεις",
            color=PALETTE["secondary"], edgecolor="white", linewidth=0.6, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(species_stats.index, rotation=45, ha="right")
    ax1.set_ylabel("Αριθμός δειγμάτων")
    ax1.set_title(f"Απόδοση ανά Είδος — {cfg.model_label}", loc="left")
    add_subtitle(ax1, f"διαχωρισμός: {split_name}", y_pad=1.10)
    ax1.margins(y=0.15)
    style_axes(ax1, hide_grid_x=True)
    ax1.legend(loc="upper left", bbox_to_anchor=(0, 1.16), ncol=2, frameon=False)

    # Secondary axis: percentage correct.
    ax2 = ax1.twinx()
    ax2.plot(x, species_stats["pct_correct"], color=PALETTE["dark"], marker="o",
             markersize=5, linewidth=1.8, label="% σωστά", zorder=4)
    ax2.set_ylabel("% σωστά")
    ax2.set_ylim(0, 115)
    ax2.grid(False)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    for xi, pct in zip(x, species_stats["pct_correct"]):
        ax2.annotate(f"{pct:.0f}%", (xi, pct), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=PALETTE["dark"])

    fig.tight_layout()
    _save(fig, cfg, f"per_species_performance_{split_name}.png")


def save_roc_curve(labels, probs, split_name: str, cfg: Config) -> float:
    """ROC curve with AUC in the legend. Returns the AUC."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(fpr, tpr, label=f"Καμπύλη ROC (AUC = {roc_auc:.3f})",
            color=PALETTE["primary"], linewidth=2.6, solid_capstyle="round", zorder=3)
    ax.fill_between(fpr, tpr, color=PALETTE["primary"], alpha=0.12, zorder=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"],
            linewidth=1.4, label="Τυχαίο", zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Ποσοστό Ψευδώς Θετικών"); ax.set_ylabel("Ποσοστό Αληθώς Θετικών")
    ax.set_title(f"Καμπύλη ROC — {cfg.model_label}", loc="left")
    add_subtitle(ax, f"διαχωρισμός: {split_name}", y_pad=1.03)
    ax.legend(loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    _save(fig, cfg, f"roc_curve_{split_name}.png")
    return roc_auc


def save_kfold_summary(df: pd.DataFrame, cfg: Config) -> None:
    """One bar (accuracy) + one marker (AUC) per species from experiment 1."""
    if df.empty:
        print("No per-species models were trained; skipping summary plot.")
        return
    df = df.sort_values("n_total", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    k = cfg.n_folds

    fig, ax1 = plt.subplots(figsize=(max(10, len(df) * 0.7), 7.5))
    ax1.bar(x, df["accuracy"] * 100, color=PALETTE["primary"], edgecolor="white",
            linewidth=0.6, zorder=3, label="Ακρίβεια (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{sp}\n(n={n})" for sp, n in zip(df["species"], df["n_total"])],
                        rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel(f"Ακρίβεια {k}-fold CV (%)")
    ax1.set_ylim(0, 118)
    ax1.set_title(f"Σύνοψη ανά Είδος ({k}-fold CV, ένα μοντέλο ανά είδος) — {cfg.model_label}",
                  loc="left")
    add_subtitle(ax1, f"Κάθε στήλη = ένα είδος, προβλέψεις out-of-fold σε {k}-fold CV",
                 y_pad=1.10)
    style_axes(ax1, hide_grid_x=True)

    ax2 = ax1.twinx()
    ax2.plot(x, df["auc"], color=PALETTE["dark"], marker="o", markersize=5,
             linewidth=1.8, label="AUC", zorder=4)
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0, 1.18)
    ax2.grid(False)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    for xi, acc, roc_v in zip(x, df["accuracy"], df["auc"]):
        ax1.annotate(f"{acc * 100:.0f}%", (xi, acc * 100), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7.5, color=PALETTE["dark"])
        ax2.annotate(f"AUC {roc_v:.2f}", (xi, roc_v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=7.5, color=PALETTE["muted"])

    # Merge the legends of both y-axes into one.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0, 1.14),
               ncol=2, frameon=False)

    fig.tight_layout()
    _save(fig, cfg, "per_species_kfold_summary.png")


def save_species_distribution_donut(species, cfg: Config) -> None:
    """Donut chart of each species' share of the dataset, total in the centre."""
    counts = pd.Series(species).value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    n = len(counts)
    pct = 100 * counts.values / total

    # Use the brand palette when it is enough, otherwise a larger colormap.
    base_colors = list(PALETTE.values())
    if n <= len(base_colors):
        colors = base_colors[:n]
    else:
        cmap = plt.get_cmap("tab20" if n <= 20 else "nipy_spectral")
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(11, 9))
    wedges, _ = ax.pie(
        counts.values, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2.5),
    )

    # Percentage labels, only on slices wide enough to read.
    for wedge, p in zip(wedges, pct):
        if p >= 3:
            ang = np.radians((wedge.theta2 + wedge.theta1) / 2)
            ax.annotate(f"{p:.1f}%", xy=(np.cos(ang) * 0.81, np.sin(ang) * 0.81),
                        ha="center", va="center", fontsize=9.5, fontweight="bold",
                        color="white")

    ax.text(0, 0.08, f"{total:,}", ha="center", va="center",
            fontsize=32, fontweight="bold", color=PALETTE["dark"])
    ax.text(0, -0.10, "συνολικά δείγματα", ha="center", va="center",
            fontsize=12, color=PALETTE["muted"])
    ax.text(0, -0.24, f"σε {n} είδη", ha="center", va="center",
            fontsize=10, color=PALETTE["muted"])

    ax.set_title("Κατανομή Ειδών στο Σύνολο Δεδομένων", loc="center", pad=18)
    ax.legend(
        wedges,
        [f"{sp.replace('_', ' ')}  —  {p:.1f}%  (n={c})"
         for sp, p, c in zip(counts.index, pct, counts.values)],
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False,
        title="Είδος",
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, cfg, "species_distribution_donut.png")
