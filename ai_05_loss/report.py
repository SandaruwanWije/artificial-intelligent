"""The lines the program prints, rounded the way the article rounds them."""

from collections.abc import Sequence

from .examples import Example
from .landscape import Spot
from .losses import Loss, SquaredError
from .models import Line, Number


def two_places(value: Number) -> str:
    return f"{round(float(value), 2):g}"


def two_figures(value: Number) -> str:
    return f"{float(value):.2g}"


def percent(share: float) -> str:
    return f"{share * 100:g}%"


def gap_lines(squared: SquaredError, answer: float, truth: float) -> list[str]:
    miss = two_places(squared.miss(answer, truth))
    return [
        f"answer {two_places(answer)}, truth {truth}",
        f"miss {truth} - {two_places(answer)} = {miss}",
        f"loss {miss} x {miss} = {two_places(squared.cost(answer, truth))}",
    ]


def fit_line(line: Line, points: Sequence[Example], loss: Loss) -> str:
    return f"mean squared error {two_places(loss.average(line, points))}"


def worst_misses_line(line: Line, points: Sequence[Example], squared: SquaredError) -> str:
    def miss_and_cost(point: Example) -> str:
        cost = squared.cost(line.answer(point.x), point.truth)
        return f"{line.miss(point):+.2f} costs {two_places(cost)}"

    above, below = max(points, key=line.miss), min(points, key=line.miss)
    return f"worst above {miss_and_cost(above)}, worst below {miss_and_cost(below)}"


def both_losses_line(losses: Sequence[Loss], answer: float, truth: float) -> str:
    return ", ".join(f"{loss.name} {two_figures(loss.cost(answer, truth))}" for loss in losses)


def surprise_line(bet: float, mood: str, cross: Loss) -> str:
    return f"bet {percent(bet)} on yes  surprise {two_figures(cross.cost(bet, 1))}  {mood}"


def share_line(share: float, cross: Loss) -> str:
    return f"{percent(share)}  costs {two_places(cross.cost(share, 1))}"


def spot_line(spot: Spot, note: str = "") -> str:
    return f"{spot.model}  loss {two_places(spot.loss)}{note}"
