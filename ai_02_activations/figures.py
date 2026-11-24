"""The pictures for the first two chapters: the step, the sigmoid and the four curves."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from common.theme import COLORS, panels, quiet, save

from .activations import Activation, Sigmoid
from .report import flatness

Z_RANGE = (-4, 4)
ZS = np.linspace(*Z_RANGE, 401)


def draw_slope(ax: Axes, curve: Activation) -> None:
    ax.plot(ZS, [curve.slope(z) for z in ZS], color=COLORS.error, linestyle="--",
            linewidth=1.6, label="slope")


def shade_almost_flat(ax: Axes, curve: Activation) -> None:
    flat = [curve.is_almost_flat(z) for z in ZS]
    ax.fill_between(ZS, 0, 1, where=flat, transform=ax.get_xaxis_transform(),
                    color=COLORS.error, alpha=0.10, linewidth=0)


def label_axes(ax: Axes) -> None:
    ax.axhline(0, color=COLORS.grid, linewidth=1)
    ax.set(xlim=Z_RANGE, xlabel="z, the weighted sum")
    quiet(ax)


def draw_step(ax: Axes) -> None:
    ax.plot([Z_RANGE[0], 0], [0, 0], color=COLORS.class_0, linewidth=2.4, label="output")
    ax.plot([0, Z_RANGE[1]], [1, 1], color=COLORS.class_0, linewidth=2.4)
    ax.plot([0, 0], [0, 1], color=COLORS.class_0, linewidth=1, linestyle=":")
    ax.scatter([0], [1], s=50, facecolors=COLORS.background, edgecolors=COLORS.class_0, zorder=3)
    ax.plot(ZS[ZS != 0], np.zeros(len(ZS) - 1), color=COLORS.error, linestyle="--",
            linewidth=1.6, label="slope")
    ax.annotate("the jump:\nno slope at all", xy=(0, 0.5), xytext=(0.5, 0.45),
                color=COLORS.error, fontsize=10)
    ax.set_title("Step: the slope is 0 everywhere")
    ax.set(ylim=(-0.15, 1.2))
    label_axes(ax)
    ax.legend(loc="upper left")


def draw_sigmoid(ax: Axes, curve: Activation) -> None:
    ax.plot(ZS, [curve.value(z) for z in ZS], color=COLORS.accent, linewidth=2.4, label="output")
    draw_slope(ax, curve)
    for z in (-2, 0, 2):
        output = curve.value(z)
        ax.scatter([z], [output], s=40, color=COLORS.accent, zorder=3)
        ax.annotate(f"{output:.2g}", (z, output), xytext=(-6, 8), textcoords="offset points",
                    ha="right", color=COLORS.accent, fontsize=10)
    peak = curve.slope(0)
    ax.scatter([0], [peak], s=40, color=COLORS.error, zorder=3)
    ax.annotate(f"slope {peak:g}", (0, peak), xytext=(10, 6), textcoords="offset points",
                va="bottom", color=COLORS.error, fontsize=10)
    ax.set_title("Sigmoid: a slope at every z")
    ax.set(ylim=(-0.15, 1.2))
    label_axes(ax)
    ax.legend(loc="upper left")


def draw_step_to_sigmoid(path: Path) -> Path:
    fig, (step_ax, sigmoid_ax) = panels(2, width=5.0, height=3.8)
    draw_step(step_ax)
    draw_sigmoid(sigmoid_ax, Sigmoid())
    return save(fig, path)


def draw_four_curves(curves: Sequence[Activation], path: Path) -> Path:
    fig, axes = panels(len(curves), width=3.6, height=3.8)
    for ax, curve in zip(axes, curves):
        shade_almost_flat(ax, curve)
        ax.plot(ZS, [curve.value(z) for z in ZS], color=COLORS.accent, linewidth=2.4,
                label=curve.name)
        draw_slope(ax, curve)
        ax.set_title(f"{curve.name}\n{flatness(curve, far=3)}", fontsize=12)
        label_axes(ax)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("output, and its slope")
    return save(fig, path)
