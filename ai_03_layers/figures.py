"""The pictures of XOR's corners and of the tents, in the blog's colours."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from common.theme import COLORS, panels, quiet, save

from .network import Network
from .neuron import Neuron
from .tents import NEURONS_PER_LAYER, DeepOrWide, deep_tents
from .xor import CORNERS, OUTPUT_LINE, Corner, wrong_corners

EDGE = (-0.45, 1.45)


def shade(ax: Axes, values: np.ndarray, grid: tuple[np.ndarray, np.ndarray], level: float) -> None:
    low, high = min(values.min(), level) - 1, max(values.max(), level) + 1
    ax.contourf(*grid, values, levels=[low, level, high],
                colors=[COLORS.class_0, COLORS.accent], alpha=0.12)
    if values.min() < level < values.max():
        ax.contour(*grid, values, levels=[level], colors=[COLORS.accent], linewidths=2)


def draw_corners(ax: Axes, spots: list[tuple[float, float]], wrong: Sequence[Corner] = ()) -> None:
    for corner, (x, y) in zip(CORNERS, spots):
        ax.scatter(x, y, s=240, zorder=3,
                   c=COLORS.accent if corner.wants else COLORS.class_0,
                   edgecolors=COLORS.error if corner in wrong else COLORS.background,
                   linewidths=3)


def input_grid(edge: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.meshgrid(np.linspace(*edge, 300), np.linspace(*edge, 300)))


def square(ax: Axes, title: str, color: str, labels: tuple[str, str] = ("x₁", "x₂")) -> None:
    ax.set_title(title, color=color)
    ax.set(xlim=EDGE, ylim=EDGE, xticks=[0, 1], yticks=[0, 1],
           xlabel=labels[0], ylabel=labels[1], aspect="equal")


def draw_one_line(tries: tuple[Neuron, ...], network: Network, path: Path) -> Path:
    fig, axes = panels(len(tries) + 1, width=3.0, height=3.4)
    grid = input_grid(EDGE)
    inputs = [corner.inputs for corner in CORNERS]
    for ax, neuron in zip(axes, tries):
        shade(ax, neuron.weighted_sum(grid), grid, level=0)
        wrong = wrong_corners(neuron)
        draw_corners(ax, inputs, wrong)
        square(ax, f"one line · {4 - len(wrong)} of 4", COLORS.error)
    shade(axes[-1], network.answer(grid), grid, level=OUTPUT_LINE)
    draw_corners(axes[-1], inputs)
    square(axes[-1], "two lines · 4 of 4", COLORS.accent)
    return save(fig, path)


def draw_new_numbers(network: Network, path: Path) -> Path:
    fig, (before, after) = panels(2, width=4.3, height=4.5)
    grid = input_grid(EDGE)
    shade(before, network.answer(grid), grid, level=OUTPUT_LINE)
    draw_corners(before, [corner.inputs for corner in CORNERS])
    square(before, "the corners: two lines", COLORS.ink)

    moved = [network.new_numbers(corner.inputs) for corner in CORNERS]
    new_grid = tuple(np.meshgrid(np.linspace(-0.4, 2.4, 300), np.linspace(-0.4, 2.4, 300)))
    shade(after, network.output_neuron.weighted_sum(new_grid), new_grid, level=OUTPUT_LINE)
    draw_corners(after, moved)
    notes = {(0, 0): ("was (0, 0)", 20), (1, 0): ("was (0, 1) and (1, 0)", -20), (2, 1): ("was (1, 1)", 20)}
    for (h1, h2), (note, offset) in notes.items():
        after.annotate(note, (h1, h2), xytext=(0, offset), textcoords="offset points",
                       ha="center", va="center", fontsize=10, color=COLORS.muted)
    after.set_title("their new numbers: one line", color=COLORS.ink)
    after.set(xlim=(-0.4, 2.4), ylim=(-0.4, 2.4), xticks=[0, 1, 2], yticks=[0, 1, 2],
              xlabel="h₁", ylabel="h₂", aspect="equal")
    return save(fig, path)


def draw_tents(rows: list[DeepOrWide], path: Path) -> Path:
    fig, axes = panels(len(rows) + 1, width=3.2, height=3.3)
    x = np.linspace(0, 1, 1025)
    for ax, row in zip(axes, rows):
        ax.plot(x, deep_tents(row.layers).answer((x,)), color=COLORS.accent, linewidth=2)
        plural = "s" if row.layers > 1 else ""
        ax.set_title(f"{row.layers} layer{plural} · {row.corners} corner{'s' if row.corners > 1 else ''}")
        ax.set(xlim=(0, 1), ylim=(0, 1.15), yticks=[0, 1], xlabel="x")
        ax.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])
        quiet(ax)
    axes[0].set_ylabel("output")

    last = rows[-1]
    bars = axes[-1]
    counts = [last.deep_neurons, last.wide_neurons]
    names = [f"deep {last.layers} × {NEURONS_PER_LAYER}", f"wide 1 × {last.wide_neurons}"]
    bars.bar(names, counts, width=0.55, color=[COLORS.accent, COLORS.error])
    for i, count in enumerate(counts):
        bars.text(i, count + 0.3, f"{count} neurons", ha="center", va="bottom", fontsize=10)
    bars.set_title(f"neurons for {last.corners} corners")
    bars.set(ylim=(0, max(counts) * 1.25), yticks=[])
    quiet(bars, grid="")
    return save(fig, path)
