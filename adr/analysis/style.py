"""Plot style for report figures: one ACL column (about 3.3 in), Times, colour-blind safe."""

from __future__ import annotations

ACL_COLUMN_IN = 3.3
INK, MUTED, GRID = "#222222", "#555555", "#DDDDDD"
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply(plt) -> None:
    """Set the report rcParams on a matplotlib.pyplot module."""
    plt.rcParams.update(RC)


def tidy_axes(ax) -> None:
    """Recessive axes: no top/right spines, thin muted left/bottom spines and ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, width=0.6, length=2.5)
