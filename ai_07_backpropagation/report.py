"""The lines part 1 prints, and the number formats both parts share."""

from .backprop import Link
from .network import Example, ForwardPass, Parameters
from .nudge import Nudge


def short(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def signed(slope: float) -> str:
    return "0" if abs(slope) < 0.005 else f"{slope:+.2f}"


def loss_text(loss: float) -> str:
    return f"{loss:.4f}".rstrip("0")


def partial(link: Link) -> str:
    return f"d({link.top})/d({link.bottom})"


def mango_lines(mango: Example, passed: ForwardPass) -> list[str]:
    return [
        f"a mango goes in: x1 {short(mango.x1)}, x2 {short(mango.x2)}",
        f"hidden layer: h1 {short(passed.h1)}, h2 {short(passed.h2)}, h3 {short(passed.h3)}",
        f"answer y {passed.y:.2f}, right answer {short(mango.right_answer)}, loss {passed.loss:.2f}",
    ]


def parameter_line(parameters: Parameters) -> str:
    names = parameters.names()
    weights = [name for name in names if name.startswith("w")]
    biases = [name for name in names if name.startswith("b")]
    return f"it can change {len(weights)} weights and {len(biases)} biases: {len(names)} parameters"


def road_line(nudge: Nudge, value: float) -> str:
    road = nudge.road()
    if "loss" not in road:
        road.append("stops")
    return f"{nudge.name}  {short(value):>4}  {' -> '.join(road):38} loss {nudge.loss_moves}"


def nudge_lines(nudge: Nudge, value: float) -> list[str]:
    before, after, change = nudge.before, nudge.after, nudge.change_in_loss
    return [
        f"nudge {nudge.name} by {short(nudge.size)}: {short(value)} -> {short(value + nudge.size)}",
        "            before    after",
        f"h1          {short(before.h1):10}{short(after.h1)}",
        f"y sum       {short(before.y_sum, 3):10}{short(after.y_sum, 3)}",
        f"answer y    {before.y:<10.2f}{after.y:.3f}",
        f"loss        {before.loss:<10.4f}{after.loss:.4f}",
        f"the loss changed by {change:.4f}, so the slope is "
        f"{change:.4f} / {short(nudge.size)} = {signed(nudge.slope)}",
    ]


def formula_lines(name: str, road: list[Link]) -> list[str]:
    return [f"the formula for {name}, one link at a time:",
            "  " + " x ".join(partial(link) for link in road),
            "every road ends with " + " x ".join(partial(link) for link in road[-2:])]


def road_length_line(roads: dict[str, list[Link]]) -> str:
    return "links on each road: " + ", ".join(f"{name} {len(road)}" for name, road in roads.items())
