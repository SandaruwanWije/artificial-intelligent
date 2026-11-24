"""Article 7's network as a picture: a circle for each part, a line for each wire."""

import math

from matplotlib.axes import Axes
from matplotlib.patches import Circle

from common.theme import COLORS

from .network import FEEDS

SPOTS = {
    "x1": (0.0, 2.1), "x2": (0.0, 0.9),
    "h1": (1.9, 2.7), "h2": (1.9, 1.5), "h3": (1.9, 0.3),
    "y": (3.5, 1.5), "loss": (4.7, 1.5),
}
RADIUS = 0.34
SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
PLATE = {"boxstyle": "round,pad=0.25", "facecolor": COLORS.background, "edgecolor": "none"}


def pretty(name: str) -> str:
    return name.translate(SUBSCRIPTS)


def minus(number: str) -> str:
    return number.replace("-", "−")


def blank_canvas(ax: Axes) -> None:
    ax.set(xlim=(-0.6, 5.3), ylim=(-0.35, 3.35), aspect="equal")
    ax.axis("off")


def draw_wire(ax: Axes, start: str, end: str, color: str, width: float = 1.5,
              dotted: bool = False) -> None:
    (x0, y0), (x1, y1) = SPOTS[start], SPOTS[end]
    length = math.dist((x0, y0), (x1, y1))
    dx, dy = (x1 - x0) / length * RADIUS, (y1 - y0) / length * RADIUS
    ax.plot([x0 + dx, x1 - dx], [y0 + dy, y1 - dy], color=color, linewidth=width,
            linestyle=":" if dotted else "-", solid_capstyle="round", zorder=1)


def label_wire(ax: Axes, name: str, text: str, color: str) -> None:
    feed = FEEDS[name]
    (x0, y0), (x1, y1) = SPOTS[feed.multiplied_by], SPOTS[feed.neuron]
    along = 0.5 if feed.neuron == "y" else 0.64
    ax.text(x0 + (x1 - x0) * along, y0 + (y1 - y0) * along, text, color=color,
            ha="center", va="center", fontsize=10, bbox=PLATE, zorder=3)


def label_bias(ax: Axes, name: str, text: str, color: str) -> None:
    x, y = SPOTS[FEEDS[name].neuron]
    ax.text(x, y - RADIUS - 0.08, text, color=color, ha="center", va="top",
            fontsize=10, bbox=PLATE, zorder=3)


def draw_part(ax: Axes, part: str, inside: str, color: str = COLORS.ink,
              ring: str = COLORS.muted) -> None:
    x, y = SPOTS[part]
    ax.add_patch(Circle((x, y), RADIUS, facecolor=COLORS.background, edgecolor=ring,
                        linewidth=1.6, zorder=2))
    ax.text(x, y + RADIUS + 0.08, pretty(part), color=COLORS.ink, ha="center", va="bottom",
            fontsize=12, fontweight="bold", zorder=3)
    ax.text(x, y, inside, color=color, ha="center", va="center", fontsize=10,
            linespacing=1.4, zorder=3)
