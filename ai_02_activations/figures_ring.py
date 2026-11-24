"""The picture for the ring: the same network's answer with each activation."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from common.theme import COLORS, panels, save

from .activations import ReLU
from .network import TinyNetwork
from .ring import Point
from .trainer import TrainingRun

EDGE = (-1.6, 1.6)
GRID = np.linspace(*EDGE, 160)


def shade_answer(ax: Axes, network: TinyNetwork) -> None:
    xx, yy = np.meshgrid(GRID, GRID)
    answer = np.vectorize(network.predict)(xx, yy)
    ax.contourf(xx, yy, answer, levels=[0, 0.5, 1], colors=[COLORS.class_0, COLORS.accent],
                alpha=0.14)
    ax.contour(xx, yy, answer, levels=[0.5], colors=[COLORS.accent], linewidths=2)


def draw_corners(ax: Axes, network: TinyNetwork) -> None:
    xx, yy = np.meshgrid(GRID, GRID)
    for neuron in network.hidden:
        weighted_sum = np.vectorize(neuron.weighted_sum)(xx, yy)
        ax.contour(xx, yy, weighted_sum, levels=[0], colors=[COLORS.ink],
                   linewidths=1, linestyles="--", alpha=0.7)


def draw_points(ax: Axes, points: Sequence[Point]) -> None:
    for label, color in ((0, COLORS.class_0), (1, COLORS.accent)):
        chosen = [p for p in points if p.label == label]
        ax.scatter([p.x for p in chosen], [p.y for p in chosen], s=14, color=color,
                   edgecolors=COLORS.background, linewidths=0.5, zorder=3)


def draw_ring(points: Sequence[Point], runs: Sequence[TrainingRun], path: Path) -> Path:
    fig, axes = panels(len(runs), width=3.8, height=4.1)
    for ax, run in zip(axes, runs):
        network = run.network
        shade_answer(ax, network)
        if isinstance(network.activation, ReLU):
            draw_corners(ax, network)
        draw_points(ax, points)
        percent = run.percent_right()
        ax.set_title(f"{network.activation.name}  ·  {percent}% right",
                     color=COLORS.ink if percent == 100 else COLORS.error)
        ax.set(xlim=EDGE, ylim=EDGE, xticks=[-1, 0, 1], yticks=[-1, 0, 1], aspect="equal")
    return save(fig, path)
