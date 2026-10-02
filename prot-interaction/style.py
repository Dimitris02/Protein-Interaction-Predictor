"""Shared matplotlib look-and-feel.

Importing this module switches matplotlib to the non-interactive "Agg"
backend, so figures are only ever written to disk and never block on a display.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after use())

PALETTE = {
    "primary": "#6C5CE7",    # indigo
    "secondary": "#00B894",  # teal green
    "accent": "#FDCB6E",     # amber
    "danger": "#E17055",     # coral
    "info": "#0984E3",       # blue
    "muted": "#636E72",      # slate gray
    "dark": "#2D3436",
}


def apply_style() -> None:
    """Install the project-wide matplotlib defaults (call once at start-up)."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "#F7F7FB",
        "axes.edgecolor": "#B9B9C6",
        "axes.linewidth": 1.0,
        "axes.labelcolor": PALETTE["dark"],
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlecolor": PALETTE["dark"],
        "axes.titlepad": 20,
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DCDCE6",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "xtick.color": PALETTE["dark"],
        "ytick.color": PALETTE["dark"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    })


def style_axes(ax, hide_grid_x: bool = False) -> None:
    """Remove top/right spines, soften the rest, and keep gridlines minimal."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#B9B9C6")
    ax.tick_params(length=0)
    if hide_grid_x:
        ax.grid(axis="x", visible=False)
    else:
        ax.grid(axis="y", alpha=0.7)


def add_subtitle(ax, text: str, y_pad: float = 1.0) -> None:
    """Small gray subtitle placed just under the main title."""
    ax.text(0.5, y_pad, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=PALETTE["muted"])
