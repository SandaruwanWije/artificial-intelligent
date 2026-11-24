"""The pictures for the landscape chapter: a valley, two maps, and the ground in 3D."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from common.theme import COLORS, panels, quiet, save

from .figures import color_of
from .landscape import Landscape, Spot
from .losses import Loss
from .models import Line, Model, Rule
from .report import two_places

LOOK_AT = {"squared error": "almost flat", "cross entropy": "a steep wall"}
BESIDE = {"above": ((0, 9), "center", "bottom"), "below": ((0, -12), "center", "top"),
          "right": ((10, 0), "left", "center")}
RINGS = {"squared error": [0.2, 0.5, 1, 2, 4, 8, 16], "cross entropy": [0.4, 0.6, 1, 1.5, 2, 3, 4]}


def paint(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(color, [COLORS.background, color])


def draw_valley(ax: Axes, row: Landscape) -> None:
    slopes, _, losses = row.heights()
    ax.plot(slopes[0], losses[0], color=COLORS.error, linewidth=2.4)
    bottom = row.lowest()
    ax.scatter(bottom.model.slope, bottom.loss, s=60, color=COLORS.accent, zorder=3)
    ax.annotate(f"best slope {two_places(bottom.model.slope)}", (bottom.model.slope, bottom.loss),
                xytext=(0, 90), textcoords="offset points", ha="center", color=COLORS.accent, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": COLORS.accent, "linestyle": ":"})
    for slope, words, offset, side in ((0.25, "too flat", 10, "left"), (1.5, "too steep", -10, "right")):
        ax.annotate(words, (slope, row.loss_at(slope, row.up[0])), xytext=(offset, 0), textcoords="offset points",
                    ha=side, va="center", color=COLORS.muted)
    ax.set_title(f"Crossing held at {two_places(row.up[0])}: a valley")
    ax.set(xlabel="slope", ylabel="loss", xlim=(row.across[0], row.across[-1]), ylim=(0, 4.5))
    quiet(ax)


def draw_map(ax: Axes, land: Landscape, names: tuple[str, str], title: str) -> None:
    across, up, losses = land.heights()
    rings = RINGS[land.loss.name]
    ax.contourf(across, up, losses, levels=[0, *rings, losses.max()], cmap=paint(color_of(land.loss)))
    lines = ax.contour(across, up, losses, levels=rings, colors=COLORS.ink, linewidths=0.6, alpha=0.6)
    ax.clabel(lines, fontsize=8, colors=COLORS.ink)
    ax.set_title(title)
    ax.set(xlabel=f"{names[0]} →", ylabel=f"{names[1]} →")
    quiet(ax, grid="")


def mark(ax: Axes, spot: Spot, color: str, words: str, beside: str = "above") -> None:
    across, up = spot.model.numbers
    offset, ha, va = BESIDE[beside]
    ax.scatter(across, up, s=60, color=color, edgecolors=COLORS.background, linewidths=1.5, zorder=4)
    ax.annotate(words, (across, up), xytext=offset, textcoords="offset points", ha=ha, va=va,
                color=color, fontsize=10, fontweight="bold")


def draw_landscape(line_map: Landscape, lines: Sequence[Line], best: Line,
                   rule_map: Landscape, wall: Rule, path: Path) -> Path:
    fig, (valley, line_ground, rule_ground) = panels(3, width=4.6, height=4.6)
    draw_valley(valley, line_map.row_at(best.crossing))

    draw_map(line_ground, line_map, ("slope", "crossing"), "Every spot is one line: a bowl")
    for line in lines:
        spot = line_map.at(line)
        near_the_top = line.crossing > 1.5
        mark(line_ground, spot, COLORS.class_0, f"loss {two_places(spot.loss)}", "below" if near_the_top else "above")
    mark(line_ground, line_map.at(best), COLORS.accent, f"lowest {two_places(line_map.at(best).loss)}", "below")

    draw_map(rule_ground, rule_map, ("weight", "bias"), "Every spot is one yes or no rule")
    mark(rule_ground, rule_map.at(wall), COLORS.class_0, "on the wall:\nsure and wrong", "right")
    mark(rule_ground, rule_map.lowest(), COLORS.accent, "the floor")
    return save(fig, path)


def draw_ground(rule_maps: dict[Loss, Landscape], room_height: float, start: Model, path: Path) -> Path:
    with plt.rc_context({"grid.color": COLORS.grid, "axes.edgecolor": COLORS.grid}):
        fig = plt.figure(figsize=(11.5, 5.2))
        for place, (loss, land) in enumerate(rule_maps.items(), start=1):
            ax = fig.add_subplot(1, 2, place, projection="3d", computed_zorder=False)
            weights, biases, losses = land.heights()
            # Cross entropy has no ceiling, so its ground is cut at the top of the room, as in the article
            shown = np.minimum(losses, room_height)
            ax.plot_surface(weights, biases, shown, cmap=paint(color_of(loss)), rcount=70, ccount=70,
                            linewidth=0, antialiased=True, vmin=0, vmax=shown.max(), zorder=1)
            spot = land.at(start)
            ax.scatter(*spot.model.numbers, spot.loss, s=60, color=COLORS.ink, depthshade=False, zorder=5)
            ax.text(*spot.model.numbers, spot.loss + 0.12 * room_height, "sure and wrong", color=COLORS.ink, zorder=6)
            ax.set_title(f"{loss.name.capitalize()}: {LOOK_AT[loss.name]}", color=color_of(loss))
            ax.set(xlabel="weight", ylabel="bias", zlim=(0, room_height))
            ax.set_zlabel("loss", labelpad=10)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color(COLORS.background)
            ax.view_init(elev=28, azim=-125)
        return save(fig, path)
