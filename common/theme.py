"""The blog's colours: white ink on black, purple for answer 1, orange for errors and slopes."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

plt.switch_backend("Agg")


@dataclass(frozen=True)
class Palette:
    background: str = "#0a0a0a"
    ink: str = "#ececec"
    muted: str = "#9a9a9a"
    grid: str = "#242424"
    class_0: str = "#d0d0d8"
    accent: str = "#a78bfa"
    error: str = "#f97316"


COLORS = Palette()

plt.rcParams.update({
    "figure.facecolor": COLORS.background,
    "axes.facecolor": COLORS.background,
    "savefig.facecolor": COLORS.background,
    "axes.edgecolor": COLORS.grid,
    "axes.labelcolor": COLORS.muted,
    "axes.titlecolor": COLORS.ink,
    "xtick.color": COLORS.muted,
    "ytick.color": COLORS.muted,
    "text.color": COLORS.ink,
    "legend.labelcolor": COLORS.ink,
    "legend.frameon": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def panels(count: int, width: float = 4.0, height: float = 4.0) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(1, count, figsize=(width * count, height), squeeze=False)
    return fig, list(axes[0])


def quiet(ax: Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=COLORS.grid, linewidth=1)
        ax.set_axisbelow(True)


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
