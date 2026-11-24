"""The one-weight pictures: the slope as a tangent, and the steps down the valley."""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from common.theme import COLORS, panels, quiet, save

from .descent import ValleyDescent
from .valley import Valley

WEIGHTS_SHOWN = (-3.0, 3.0)
LOSSES_SHOWN = (0, 5.2)


def signed(number: float) -> str:
    return f"{number:.2f}".replace("-", "−")


def draw_valley(ax: Axes, valley: Valley) -> None:
    weights = np.linspace(*WEIGHTS_SHOWN, 200)
    ax.plot(weights, [valley.loss(w) for w in weights], color=COLORS.class_0, linewidth=2)
    ax.set(xlim=WEIGHTS_SHOWN, ylim=LOSSES_SHOWN, xlabel="weight", ylabel="loss")
    quiet(ax)


def draw_slope_triangle(ax: Axes, valley: Valley, weight: float) -> None:
    loss, slope = valley.loss(weight), valley.slope(weight)
    reach = np.array([-1.2, 1.6])
    ax.plot(weight + reach, loss + slope * reach, color=COLORS.error, linewidth=1.5)
    ax.plot([weight, weight + 1], [loss, loss], color=COLORS.muted, linestyle="--")
    ax.plot([weight + 1, weight + 1], [loss, loss + slope], color=COLORS.error, linewidth=3)
    ax.text(weight + 0.6, loss + 0.2, "1 step right", ha="center", color=COLORS.muted, fontsize=10)
    ax.text(weight + 1.15, loss + slope / 2, f"change\n{signed(slope)}",
            va="center", color=COLORS.error, fontsize=10, fontweight="bold", zorder=4,
            bbox=dict(facecolor=COLORS.background, edgecolor="none", pad=1))
    ax.scatter(weight, loss, s=140, color=COLORS.accent, zorder=3)


def draw_slopes(valley: Valley, spots: tuple[float, ...], path: Path) -> Path:
    fig, axes = panels(len(spots), width=4.6, height=4.0)
    for ax, weight in zip(axes, spots):
        draw_valley(ax, valley)
        draw_slope_triangle(ax, valley, weight)
        ax.set_title(f"weight {signed(weight)}  ·  slope {signed(valley.slope(weight))}")
    return save(fig, path)


def draw_steps(valley: Valley, descent: ValleyDescent, path: Path) -> Path:
    fig, (trip_ax, sizes_ax) = panels(2, width=4.8, height=4.0)
    draw_valley(trip_ax, valley)
    spots = [step.weight for step in descent.steps] + [descent.end]
    losses = [valley.loss(weight) for weight in spots]
    for (w0, l0), (w1, l1) in zip(zip(spots, losses), zip(spots[1:], losses[1:])):
        trip_ax.annotate("", xy=(w1, l1), xytext=(w0, l0),
                         arrowprops=dict(arrowstyle="->", color=COLORS.accent, linewidth=1.5,
                                         connectionstyle="arc3,rad=-0.35"))
    trip_ax.scatter(spots, losses, s=50, color=COLORS.accent, zorder=3)
    trip_ax.set_title(f"Learning rate {descent.learning_rate}: steps downhill")

    numbers = np.arange(1, len(descent.steps) + 1)
    sizes = [step.size for step in descent.steps]
    sizes_ax.bar(numbers, sizes, width=0.6, color=COLORS.accent)
    for number, size in zip(numbers, sizes):
        sizes_ax.text(number, size + 0.04, f"{size:.2f}", ha="center", va="bottom", fontsize=10)
    sizes_ax.set(xticks=numbers, ylim=(0, 2.05), xlabel="step", ylabel="step size")
    sizes_ax.set_title("Smaller every time")
    quiet(sizes_ax)
    return save(fig, path)
