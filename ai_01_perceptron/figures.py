from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from common.theme import COLORS, panels, quiet, save

from .gates import Corner, LogicGate
from .perceptron import Perceptron, Weights
from .trainer import TrainingRun

SQUARE_WITH_MARGIN = (-0.45, 1.45)


def shade_by_answer(ax: Axes, neuron: Perceptron) -> None:
    grid = np.linspace(*SQUARE_WITH_MARGIN, 300)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.vectorize(neuron.total)(xx, yy)
    ax.contourf(xx, yy, zz, levels=[zz.min() - 1, 0, zz.max() + 1],
                colors=[COLORS.class_0, COLORS.accent], alpha=0.12)
    if zz.min() < 0 < zz.max():
        ax.contour(xx, yy, zz, levels=[0], colors=[COLORS.accent], linewidths=2)


def draw_corners(ax: Axes, neuron: Perceptron, gate: LogicGate) -> list[Corner]:
    wrong = neuron.wrong_corners(gate)
    for corner in gate.corners:
        ax.scatter(corner.x1, corner.x2, s=240, zorder=3,
                   c=COLORS.accent if corner.answer else COLORS.class_0,
                   edgecolors=COLORS.error if corner in wrong else COLORS.background,
                   linewidths=3)
    return wrong


def draw_square(ax: Axes, weights: Weights, gate: LogicGate, title: str) -> None:
    neuron = Perceptron(weights)
    shade_by_answer(ax, neuron)
    wrong = draw_corners(ax, neuron, gate)
    ax.set_title(title, color=COLORS.error if wrong else COLORS.ink)
    ax.set(xlim=SQUARE_WITH_MARGIN, ylim=SQUARE_WITH_MARGIN, xticks=[0, 1], yticks=[0, 1],
           xlabel="x₁", ylabel="x₂", aspect="equal")


def draw_passes(gate: LogicGate, run: TrainingRun, path: Path) -> Path:
    passes = zip(run.after_each_pass, run.mistakes_per_pass)
    steps = [("Start", run.start)] + [
        (f"Pass {number} · {fixes} {'fix' if fixes == 1 else 'fixes'}", weights)
        for number, (weights, fixes) in enumerate(passes, start=1)
    ]
    fig, axes = panels(len(steps), width=3.2, height=3.5)
    for ax, (title, weights) in zip(axes, steps):
        draw_square(ax, weights, gate, title)
    return save(fig, path)


def draw_boundaries(final: dict[LogicGate, Weights], path: Path) -> Path:
    fig, axes = panels(len(final), height=4.3)
    for ax, (gate, weights) in zip(axes, final.items()):
        wrong = len(Perceptron(weights).wrong_corners(gate))
        note = "learned" if wrong == 0 else f"{wrong} corners wrong"
        draw_square(ax, weights, gate, f"{gate.name}  ·  {note}")
    return save(fig, path)


def draw_mistakes(runs: dict[LogicGate, TrainingRun], path: Path) -> Path:
    longest = max(run.passes for run in runs.values())
    fig, axes = panels(len(runs), height=3.4)
    for ax, (gate, run) in zip(axes, runs.items()):
        color = COLORS.accent if run.learned else COLORS.error
        passes = np.arange(1, run.passes + 1)
        ax.bar(passes, run.mistakes_per_pass, width=0.6, color=color)
        for number, mistakes in zip(passes, run.mistakes_per_pass):
            ax.text(number, mistakes + 0.12, str(mistakes), ha="center", va="bottom", fontsize=10)
        ax.set_title(gate.name, color=COLORS.ink if run.learned else COLORS.error)
        ax.set(xlim=(0.4, longest + 0.6), ylim=(0, 4.8), xlabel="pass",
               xticks=range(1, longest + 1), yticks=range(5))
        quiet(ax)
    axes[0].set_ylabel("mistakes")
    return save(fig, path)
