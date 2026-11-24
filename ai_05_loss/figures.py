"""The pictures for squared error and cross entropy, in the blog's colours."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from common.theme import COLORS, panels, quiet, save

from .examples import Example
from .losses import Loss, SquaredError
from .models import Line
from .report import percent, two_figures, two_places

ANSWERS = np.linspace(0.005, 0.995, 400)
PLANE = (0, 5.1)


def color_of(loss: Loss) -> str:
    return COLORS.error if isinstance(loss, SquaredError) else COLORS.accent


def draw_square(ax: Axes, left: float, bottom: float, side: float) -> None:
    ax.add_patch(Rectangle((left, bottom), side, side, facecolor=COLORS.error, alpha=0.3,
                           edgecolor=COLORS.error, linewidth=1.2))


def draw_twice(ax: Axes, squared: SquaredError, small: float, big: float) -> None:
    for left, miss in ((0, small), (0.5, big)):
        draw_square(ax, left, 0, miss)
        cost = two_places(squared.cost(answer=0, truth=miss))
        ax.text(left + miss / 2, miss + 0.04, f"{miss} x {miss}\n= {cost}", ha="center", va="bottom")
    ax.plot([0.5 + small] * 2, [0, big], [0.5, 0.5 + big], [small] * 2,
            color=COLORS.error, linewidth=1, linestyle="--")
    ax.set_title("Twice the miss, four times the cost")
    ax.set(xlim=(-0.1, 1.2), ylim=(-0.1, 1.0), aspect="equal", xticks=[], yticks=[])
    ax.axis("off")


def draw_fit(ax: Axes, line: Line, points: Sequence[Example], squared: SquaredError, title: str) -> None:
    for point in points:
        draw_square(ax, point.x, min(point.truth, line.answer(point.x)), abs(line.miss(point)))
    ax.plot(PLANE, [line.answer(x) for x in PLANE], color=COLORS.accent, linewidth=2)
    ax.scatter([p.x for p in points], [p.truth for p in points], s=36, color=COLORS.class_0, zorder=3)
    ax.set_title(f"{title}\nmean squared error {two_places(squared.average(line, points))}")
    ax.set(xlim=PLANE, ylim=(0, 5), aspect="equal", xlabel="x", ylabel="y")
    quiet(ax, grid="")


def draw_squares(misses: tuple[float, float], poor: Line, best: Line, points: Sequence[Example],
                 squared: SquaredError, path: Path) -> Path:
    fig, (twice, before, after) = panels(3, width=4.4, height=5.0)
    draw_twice(twice, squared, *misses)
    draw_fit(before, poor, points, squared, "A poor line")
    draw_fit(after, best, points, squared, "The best line")
    return save(fig, path)


def label(ax: Axes, at: float, cost: float, color: str, on_the_right: bool) -> None:
    ax.annotate(two_figures(cost), (at, cost), xytext=(8 if on_the_right else -8, 6), textcoords="offset points",
                ha="left" if on_the_right else "right", va="bottom", color=color, fontweight="bold",
                bbox={"facecolor": COLORS.background, "edgecolor": "none", "pad": 1})


def mark_answer(ax: Axes, at: float, truth: int, losses: Sequence[Loss], note: str, note_at: tuple[float, float]) -> None:
    squared, cross = losses
    for loss, on_the_right in ((squared, at < 0.05), (cross, at < 0.95)):
        ax.scatter(at, loss.cost(at, truth), s=40, color=color_of(loss), zorder=3)
        label(ax, at, loss.cost(at, truth), color_of(loss), on_the_right)
    ax.annotate(note, (at, cross.cost(at, truth)), xytext=note_at, ha="center", color=COLORS.muted,
                arrowprops={"arrowstyle": "-", "color": COLORS.muted, "linestyle": ":"})


def draw_losses(squared: Loss, cross: Loss, answer: float, sure_and_wrong: float, path: Path) -> Path:
    fig, (yes, no) = panels(2, width=5.2, height=4.2)
    for ax, truth in ((yes, 1), (no, 0)):
        for loss in (squared, cross):
            ax.plot(ANSWERS, loss.cost(ANSWERS, truth), color=color_of(loss), linewidth=2.4, label=loss.name)
        ax.axhline(1, color=COLORS.muted, linewidth=1, linestyle=":")
        ax.text(0.98 if truth == 1 else 0.02, 1.08, "squared error never goes above 1",
                ha="right" if truth == 1 else "left", color=COLORS.muted, fontsize=9)
        ax.set_title(f"The truth is {'yes' if truth == 1 else 'no'}")
        ax.set(xlim=(0, 1), ylim=(0, 5.2), xlabel="the network's answer for yes")
        quiet(ax)
    mark_answer(yes, sure_and_wrong, 1, (squared, cross), "sure and wrong", (0.3, 4.3))
    mark_answer(yes, answer, 1, (squared, cross), "the network's answer", (0.25, 2.2))
    mark_answer(no, 1 - sure_and_wrong, 0, (squared, cross), "sure and wrong", (0.7, 4.3))
    yes.set_ylabel("loss")
    yes.legend(loc="upper right")
    return save(fig, path)


def draw_surprise(squared: Loss, cross: Loss, bets: Sequence[float], shares: Sequence[float], path: Path) -> Path:
    fig, (by_bet, by_share) = panels(2, width=5.8, height=4.2)
    places = np.arange(len(bets))
    for offset, loss in ((-0.2, squared), (0.2, cross)):
        costs = [loss.cost(bet, 1) for bet in bets]
        by_bet.bar(places + offset, costs, width=0.38, color=color_of(loss), label=loss.name)
        for place, cost in zip(places, costs):
            by_bet.text(place + offset, cost + 0.08, two_figures(cost), ha="center", va="bottom", fontsize=9)
    by_bet.set_title("The truth is yes: surprise for each bet")
    by_bet.set(xticks=places, xticklabels=[f"bet {percent(b)}" for b in bets], ylim=(0, 5.2), ylabel="loss")
    by_bet.legend(loc="upper left")
    quiet(by_bet)

    costs = [cross.cost(share, 1) for share in shares]
    by_share.bar(range(len(shares)), costs, width=0.6, color=COLORS.accent)
    for place, cost in enumerate(costs):
        by_share.text(place, cost + 0.05, two_places(cost), ha="center", va="bottom")
    by_share.set_title(f"Each halving adds the same {two_places(costs[1])}")
    by_share.set(xticks=range(len(shares)), xticklabels=[percent(s) for s in shares], ylim=(0, 2.5),
                 xlabel="share for the right answer", ylabel="cross entropy")
    quiet(by_share)
    return save(fig, path)
