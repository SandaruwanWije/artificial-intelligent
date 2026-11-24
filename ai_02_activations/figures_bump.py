"""The pictures for the bump: layers with no bend, then three ReLU corners that draw it."""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from common.theme import COLORS, panels, quiet, save

from .bump import XS, FinalNeuron, ReluNeuron, goal
from .layers import Layer

X_RANGE = (-1.2, 3.2)
GRID = np.linspace(*X_RANGE, 221)


def draw_goal(ax: Axes) -> None:
    ax.plot(GRID, [goal(x) for x in GRID], color=COLORS.class_0, linestyle="--",
            linewidth=1.6, label="goal: the bump")


def finish(ax: Axes, title: str, ylim: tuple[float, float]) -> None:
    ax.axhline(0, color=COLORS.grid, linewidth=1)
    ax.set_title(title)
    ax.set(xlim=X_RANGE, ylim=ylim, xticks=XS, xlabel="x, the network's input")
    quiet(ax)


def draw_line_through(ax: Axes, layer: Layer, x: float, label: str) -> None:
    ax.plot(GRID, [layer(g) for g in GRID], color=COLORS.accent, linewidth=2.4, label=label)
    ax.scatter([x], [layer(x)], s=50, color=COLORS.accent, zorder=3)
    ax.annotate(f"x = {x:g} gives {layer(x):g}", (x, layer(x)), xytext=(-10, 8),
                textcoords="offset points", ha="right", color=COLORS.ink, fontsize=10)


def draw_no_bend(layer_1: Layer, layer_2: Layer, path: Path) -> Path:
    together = layer_1.then(layer_2)
    fig, (one_ax, two_ax) = panels(2, width=4.6, height=3.8)
    for ax, layer, title in ((one_ax, layer_1, "Layer 1: a straight line"),
                             (two_ax, together, "Layers 1 and 2: still straight")):
        draw_goal(ax)
        draw_line_through(ax, layer, 2, f"w {layer.w:g}, b {layer.b:g}")
        finish(ax, title, ylim=(-1.4, 3.4))
        ax.legend(loc="upper left", fontsize=9)
    return save(fig, path)


def draw_neuron(ax: Axes, neuron: ReluNeuron) -> None:
    ax.plot(GRID, [neuron.z(x) for x in GRID], color=COLORS.muted, linestyle=":",
            linewidth=1.4, label="z, before ReLU")
    ax.plot(GRID, [neuron.gives(x) for x in GRID], color=COLORS.accent, linewidth=2.4,
            label="after ReLU")
    ax.scatter(XS, [neuron.gives(x) for x in XS], s=40, color=COLORS.accent, zorder=3)
    ax.scatter([neuron.corner], [0], s=200, facecolors="none", edgecolors=COLORS.error,
               linewidths=2, zorder=4)
    sign = "−" if neuron.bias < 0 else "+"
    finish(ax, f"{neuron.name}: z = x {sign} {abs(neuron.bias):g}", ylim=(-3.4, 3.4))


def draw_relu_bump(final: FinalNeuron, path: Path) -> Path:
    fig, axes = panels(len(final.inputs) + 1, width=3.4, height=3.8)
    for ax, neuron in zip(axes, final.inputs):
        draw_neuron(ax, neuron)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("output")

    sum_ax = axes[-1]
    draw_goal(sum_ax)
    sum_ax.plot(GRID, [final.y(x) for x in GRID], color=COLORS.accent, linewidth=2.4,
                label="y, the final neuron")
    sum_ax.scatter(XS, [final.y(x) for x in XS], s=40, color=COLORS.accent, zorder=3)
    w1, w2, w3 = final.weights
    finish(sum_ax, f"y = {w1:g}·n1 − {abs(w2):g}·n2 + {w3:g}·n3", ylim=(-3.4, 3.4))
    sum_ax.legend(loc="upper left", fontsize=9)
    return save(fig, path)
