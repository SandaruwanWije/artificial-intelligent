"""The charts: one nudge up close, the 13 formulas, three ways agreeing, and the loss falling."""

from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from common.theme import COLORS, panels, quiet, save

from .backprop import Backprop, Sweep
from .drawing import minus, pretty
from .network import Example, Network
from .nudge import Nudge
from .report import loss_text, short, signed
from .trainer import TrainingRun

FORMULA_COLUMNS = ("own link", "ReLU", "wire into y", "sigmoid", "loss", "slope")


def draw_nudge(network: Network, example: Example, nudge: Nudge, path: Path) -> Path:
    start = network.parameters.value(nudge.name)
    name = pretty(nudge.name)

    def loss_at(value: float) -> float:
        moved = network.parameters.nudged(nudge.name, value - start)
        return Network(moved).forward(example).loss

    fig, (whole, close) = panels(2, width=5, height=4.2)
    for ax, (low, high) in ((whole, (0.2, 1.4)), (close, (start - 0.004, start + 0.014))):
        values = np.linspace(low, high, 200)
        ax.plot(values, [loss_at(v) for v in values], color=COLORS.error, linewidth=2)
        ax.set(xlabel=name, xlim=(low, high))
        quiet(ax)
    near = np.array([start - 0.25, start + 0.25])
    whole.plot(near, nudge.before.loss + nudge.slope * (near - start), color=COLORS.ink,
               linewidth=1.2, linestyle="--")
    whole.text(near[1] + 0.03, nudge.before.loss + nudge.slope * 0.25, f"slope {minus(signed(nudge.slope))}",
               va="center", fontsize=10)
    whole.scatter([start], [nudge.before.loss], s=60, color=COLORS.accent, zorder=3)
    whole.set(ylabel="loss", title=f"The loss as {name} alone moves")

    after = start + nudge.size
    close.scatter([start, after], [nudge.before.loss, nudge.after.loss], s=60,
                  color=COLORS.accent, zorder=3)
    close.plot([start, after, after], [nudge.before.loss, nudge.before.loss, nudge.after.loss],
               color=COLORS.ink, linewidth=1.2, linestyle="--")
    close.text(start + nudge.size / 2, nudge.before.loss + 0.0002, f"nudge {short(nudge.size)}",
               ha="center", va="bottom", fontsize=10)
    close.text(after + 0.0004, (nudge.before.loss + nudge.after.loss) / 2,
               f"loss {minus(f'{nudge.change_in_loss:.4f}')}", va="center", fontsize=10)
    close.set(xticks=[start, after], xticklabels=[short(start), short(after)])
    close.set(title=f"Up close: {minus(f'{nudge.change_in_loss:.4f}')} ÷ {short(nudge.size)}"
                    f" = {minus(signed(nudge.slope))}")
    return save(fig, path)


def draw_formulas(backprop: Backprop, path: Path) -> Path:
    names = backprop.parameters.names()
    fig, (ax,) = panels(1, width=8.4, height=7.4)
    ax.set(xlim=(-1, len(FORMULA_COLUMNS)), ylim=(len(names), -1.6))
    ax.axis("off")
    ax.add_patch(Rectangle((2.55, -0.5), 1.9, len(names), color=COLORS.accent, alpha=0.14, linewidth=0))
    ax.text(3.5, -1.3, "the same two links, 13 times", color=COLORS.accent, ha="center", fontsize=10)
    for column, heading in enumerate(FORMULA_COLUMNS):
        ax.text(column, -0.75, heading, color=COLORS.muted, ha="center", fontsize=10)
    for row, name in enumerate(names):
        road = backprop.road(name)
        middle = road[1:3] if len(road) == 5 else [None, None]
        ax.text(-0.75, row, pretty(name), ha="left", va="center", fontsize=12, fontweight="bold")
        for column, link in enumerate([road[0], *middle, *road[-2:]]):
            if link is not None:
                ax.text(column, row, minus(short(link.ratio)), ha="center", va="center", fontsize=11,
                        color=COLORS.muted if link.ratio == 0 else COLORS.ink)
        slope = minus(signed(backprop.slope_link_by_link(name)))
        ax.text(len(FORMULA_COLUMNS) - 1, row, slope, ha="center", va="center", fontsize=11,
                fontweight="bold", color=COLORS.error)
    ax.set_title("Every formula ends with the same two links")
    return save(fig, path)


def draw_agreement(nudges: dict[str, Nudge], sweep: Sweep, path: Path) -> Path:
    names = list(sweep.slopes)
    spots = np.arange(len(names))
    nudged = [nudges[name].slope for name in names]
    swept = [sweep.slopes[name] for name in names]
    fig, (ax,) = panels(1, width=10, height=4.4)
    ax.bar(spots - 0.2, nudged, width=0.38, color=COLORS.class_0, label="nudge test")
    ax.bar(spots + 0.2, swept, width=0.38, color=COLORS.error, label="backward sweep")
    for spot, slope in zip(spots, swept):
        below = slope < -0.005
        ax.text(spot, slope - 0.02 if below else max(slope, 0) + 0.02, minus(signed(slope)),
                ha="center", va="top" if below else "bottom", fontsize=9)
    ax.axhline(0, color=COLORS.muted, linewidth=1)
    ax.set(xticks=spots, xticklabels=[pretty(name) for name in names], ylabel="slope",
           ylim=(-0.33, 0.24))
    ax.legend(loc="upper right")
    quiet(ax)
    ax.set_title("The nudge test and the backward sweep agree on all 13 slopes")
    return save(fig, path)


def draw_training(run: TrainingRun, path: Path) -> Path:
    rounds = np.arange(len(run.passes))
    fig, (losses, answers) = panels(2, width=5, height=4)
    losses.plot(rounds, run.losses, color=COLORS.error, linewidth=2)
    answers.plot(rounds, run.answers, color=COLORS.accent, linewidth=2)
    answers.axhline(1, color=COLORS.muted, linewidth=1, linestyle="--")
    answers.text(rounds[-1], 1.005, "right answer 1", ha="right", va="bottom", fontsize=10,
                 color=COLORS.muted)
    for r in (0, 1, rounds[-1]):
        last = r == rounds[-1]
        side = {"ha": "right" if last else "left", "fontsize": 10}
        losses.scatter([r], [run.losses[r]], s=40, color=COLORS.error, zorder=3)
        losses.text(r - 1 if last else r + 1, run.losses[r] + 0.006, loss_text(run.losses[r]), **side)
        answers.scatter([r], [run.answers[r]], s=40, color=COLORS.accent, zorder=3)
        answers.text(r - 1 if last else r + 1, run.answers[r] - 0.012, f"{run.answers[r]:.2f}",
                     va="top", **side)
    losses.set(title="The loss falls round after round", xlabel="round", ylabel="loss",
               ylim=(0, 0.19))
    answers.set(title="The answer climbs toward 1", xlabel="round", ylabel="answer y",
                ylim=(0.55, 1.04))
    for ax in (losses, answers):
        quiet(ax)
    return save(fig, path)
