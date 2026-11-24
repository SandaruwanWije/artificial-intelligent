"""Plots for the perceptron article, in the blog's colours.

White ink on black, purple for answer 1 and the learned line, and orange
for anything wrong, the same as the article's own figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from perceptron import Perceptron

BG = "#0a0a0a"
INK = "#ececec"
MUTED = "#9a9a9a"
GRID = "#242424"
CLASS_0 = "#d0d0d8"  # answer 0: plain ink
CLASS_1 = "#a78bfa"  # answer 1: the purple accent
WRONG = "#f97316"  # a wrong guess: orange

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def _square(ax, neuron: Perceptron, corners: list, name: str) -> None:
    """One gate: the square shaded by the answer, the line, and the four corners."""
    # total() is plain arithmetic, so it works on a whole grid of points at once
    xx, yy = np.meshgrid(np.linspace(-0.45, 1.45, 400), np.linspace(-0.45, 1.45, 400))
    zz = neuron.total(xx, yy)
    ax.contourf(xx, yy, zz, levels=[zz.min() - 1, 0, zz.max() + 1],
                colors=[CLASS_0, CLASS_1], alpha=0.12)
    ax.contour(xx, yy, zz, levels=[0], colors=[CLASS_1], linewidths=2)

    left = 0
    for x1, x2, answer in corners:
        wrong = neuron.guess(x1, x2) != answer
        left += wrong
        ax.scatter(x1, x2, s=260, c=CLASS_1 if answer else CLASS_0, zorder=3,
                   edgecolors=WRONG if wrong else BG, linewidths=3)
    note = "learned" if left == 0 else f"{left} corners wrong"
    ax.set_title(f"{name}  ·  {note}", color=INK if left == 0 else WRONG)
    ax.set(xlim=(-0.45, 1.45), ylim=(-0.45, 1.45), xticks=[0, 1], yticks=[0, 1],
           xlabel="x₁", ylabel="x₂", aspect="equal")


def plot_boundaries(neurons: dict[str, Perceptron], gates: dict[str, list], path: Path) -> None:
    """Every gate side by side: the line each perceptron ended on."""
    fig, axes = plt.subplots(1, len(neurons), figsize=(4 * len(neurons), 4.3))
    for ax, (name, neuron) in zip(axes, neurons.items()):
        _square(ax, neuron, gates[name], name)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mistakes(histories: dict[str, list[int]], path: Path) -> None:
    """Mistakes per pass, one bar chart per gate: only a learnable gate reaches 0."""
    longest = max(len(m) for m in histories.values())
    fig, axes = plt.subplots(1, len(histories), figsize=(4 * len(histories), 3.4), sharey=True)
    for ax, (name, mistakes) in zip(axes, histories.items()):
        learned = mistakes[-1] == 0
        passes = np.arange(1, len(mistakes) + 1)
        ax.bar(passes, mistakes, width=0.6, color=CLASS_1 if learned else WRONG)
        for p, m in zip(passes, mistakes):
            ax.text(p, m + 0.12, str(m), ha="center", va="bottom", color=INK, fontsize=10)
        ax.set_title(name, color=INK if learned else WRONG)
        ax.set(xlim=(0.4, longest + 0.6), ylim=(0, 4.8), xlabel="pass")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mistakes")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)
