"""The pictures of trained networks on the noisy XOR cloud."""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from common.theme import COLORS, panels, save

from .cloud import SQUARE, Point
from .trainer import Trial


def draw_trial(ax: Axes, trial: Trial, points: tuple[Point, ...]) -> None:
    grid = tuple(np.meshgrid(np.linspace(*SQUARE, 300), np.linspace(*SQUARE, 300)))
    answers = trial.network.answer(grid)
    ax.contourf(*grid, answers, levels=[0, 0.5, 1], colors=[COLORS.class_0, COLORS.accent], alpha=0.12)
    ax.contour(*grid, answers, levels=[0.5], colors=[COLORS.ink], linewidths=2)
    for line in trial.lines_switched_on():
        ax.contour(*grid, line.weighted_sum(grid), levels=[0], colors=[COLORS.muted],
                   linewidths=1.2, linestyles="dashed")

    for wants, color in ((0, COLORS.class_0), (1, COLORS.accent)):
        group = [point for point in points if point.wants == wants]
        ax.scatter([p.x1 for p in group], [p.x2 for p in group], s=14, c=color, zorder=3)
    wrong = trial.wrong_points(points)
    ax.scatter([p.x1 for p in wrong], [p.x2 for p in wrong], s=90, zorder=4,
               facecolors="none", edgecolors=COLORS.error, linewidths=1.5)

    n, score = trial.hidden_neurons, trial.percent_right(points)
    ax.set_title(f"{n} hidden neuron{'s' if n > 1 else ''} · {score}%\nstart {trial.seed}",
                 color=COLORS.ink if score == 100 else COLORS.error)
    ax.set(xlim=SQUARE, ylim=SQUARE, xticks=[-1, 0, 1], yticks=[-1, 0, 1],
           xlabel="x₁", ylabel="x₂", aspect="equal")


def draw_cloud(trials: list[Trial], points: tuple[Point, ...], path: Path) -> Path:
    fig, axes = panels(len(trials), width=4.0, height=4.5)
    for ax, trial in zip(axes, trials):
        draw_trial(ax, trial, points)
    return save(fig, path)
