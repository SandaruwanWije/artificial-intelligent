"""The two-weight pictures: the gradient at the start, and three learning rates."""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from common.theme import COLORS, panels, quiet, save

from .bowl import Bowl, Weights
from .descent import BowlDescent

W1_SHOWN = (-3.8, 3.8)
W2_SHOWN = (-1.5, 1.5)
RINGS = [0.25, 1, 2.5, 5, 8, 11]
RING_LABELS_AT = [(0.7, 0), (1.4, 0), (2.2, 0), (3.2, 0)]
RATE_COLORS = {"too small": COLORS.class_0, "about right": COLORS.accent, "too big": COLORS.error}


def draw_map(ax: Axes, bowl: Bowl, w1_shown=W1_SHOWN, w2_shown=W2_SHOWN,
             ring_labels_at=RING_LABELS_AT) -> None:
    w1, w2 = np.meshgrid(np.linspace(*w1_shown, 300), np.linspace(*w2_shown, 200))
    losses = bowl.loss(Weights(w1, w2))
    rings = ax.contour(w1, w2, losses, levels=RINGS, colors=COLORS.muted, linewidths=0.8)
    ax.clabel(rings, fontsize=8, colors=COLORS.muted, fmt="%g", manual=ring_labels_at)
    ax.scatter(0, 0, marker="+", s=80, color=COLORS.ink, zorder=3)
    ax.set(xlim=w1_shown, ylim=w2_shown, xlabel="weight w₁", ylabel="weight w₂", aspect="equal")
    quiet(ax, grid="")


def arrow(ax: Axes, start: Weights, d1: float, d2: float, color: str, label: str,
          where: tuple[float, float]) -> None:
    ax.annotate("", xy=(start.w1 + d1, start.w2 + d2), xytext=(start.w1, start.w2),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.2))
    ax.text(*where, label, color=color, fontsize=10, fontweight="bold", ha="center")


def draw_gradient(bowl: Bowl, start: Weights, learning_rate: float, path: Path) -> Path:
    fig, (ax,) = panels(1, width=5.6, height=5.4)
    draw_map(ax, bowl, w1_shown=(-4.3, -2.1), w2_shown=(-0.2, 2.0),
             ring_labels_at=[(-3.15, 0.1), (-3.9, 0.3), (-3.8, 1.15)])
    gradient = bowl.gradient(start)
    across, up = learning_rate * gradient.slope_for_w1, learning_rate * gradient.slope_for_w2
    w1_label = f"slope for w₁: {gradient.slope_for_w1:g}".replace("-", "−")
    arrow(ax, start, across, 0, COLORS.error, w1_label, (start.w1 - 0.62, start.w2 - 0.17))
    arrow(ax, start, 0, up, COLORS.error,
          f"slope for w₂: {gradient.slope_for_w2:g}", (start.w1 + 0.55, start.w2 + 0.5))
    arrow(ax, start, across, up, COLORS.ink, "gradient: uphill",
          (start.w1 - 0.48, start.w2 + 0.9))
    arrow(ax, start, -across, -up, COLORS.accent, f"step: −{learning_rate} × gradient",
          (start.w1 + 0.62, start.w2 - 0.85))
    ax.scatter(start.w1, start.w2, s=140, color=COLORS.accent, zorder=4)
    ax.set_xticks(np.arange(-4.0, -2.0, 0.5))
    ax.set_title("Two slopes make the gradient")
    ax.text(0.98, 0.02, f"arrows drawn at {learning_rate} × their size", transform=ax.transAxes,
            ha="right", va="bottom", color=COLORS.muted, fontsize=9)
    return save(fig, path)


def draw_learning_rates(bowl: Bowl, descents: dict[str, BowlDescent], path: Path) -> Path:
    fig, axes = panels(len(descents), width=4.6, height=2.6)
    for ax, (name, descent) in zip(axes, descents.items()):
        draw_map(ax, bowl)
        w1s = [spot.w1 for spot in descent.path]
        w2s = [spot.w2 for spot in descent.path]
        ax.plot(w1s, w2s, color=RATE_COLORS[name], linewidth=1.5, marker="o", markersize=3)
        ax.scatter(w1s[0], w2s[0], s=80, color=RATE_COLORS[name], zorder=4)
        ax.set_title(f"{descent.learning_rate}  ·  {name}", color=RATE_COLORS[name])
    return save(fig, path)


def draw_rate_losses(descents: dict[str, BowlDescent], path: Path) -> Path:
    fig, axes = panels(len(descents), width=4.0, height=3.2)
    highest = max(max(descent.losses) for descent in descents.values())
    for ax, (name, descent) in zip(axes, descents.items()):
        steps = np.arange(len(descent.losses))
        ax.plot(steps, descent.losses, color=RATE_COLORS[name], linewidth=2)
        ax.text(steps[-1], descent.losses[-1] + 0.5, f"{round(descent.losses[-1], 3):g}",
                ha="right", va="bottom", color=RATE_COLORS[name], fontsize=10, fontweight="bold")
        ax.set(xlim=(0, steps[-1]), ylim=(0, highest * 1.1), xlabel="step")
        ax.set_title(f"{descent.learning_rate}  ·  {name}", color=RATE_COLORS[name])
        quiet(ax)
    axes[0].set_ylabel("loss")
    return save(fig, path)
