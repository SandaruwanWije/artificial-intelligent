"""The lines part 2 prints: the links, the sweep, the step and the rounds."""

from .backprop import Backprop, Link, Sweep
from .network import ForwardPass
from .report import loss_text, partial, short, signed
from .trainer import TrainingRun


def link_lines(road: list[Link]) -> list[str]:
    return [f"{partial(link):18}{short(link.ratio):7}{link.why}" for link in road]


def shortcut_lines(backprop: Backprop) -> list[str]:
    y, miss = backprop.passed.y, backprop.miss()
    sigmoid_link, loss_link = backprop.link_onward_from["y sum"], backprop.link_onward_from["y"]
    return [
        f"sigmoid's shortcut: y x (1 - y) = {y:.2f} x {short(-miss)} = {short(sigmoid_link.ratio)}",
        f"the loss's link: 2 x the miss = 2 x {short(miss)} = {short(loss_link.ratio)}",
    ]


def product_line(name: str, road: list[Link], slope: float) -> str:
    return f"{name}  {' x '.join(short(link.ratio) for link in road)} = {signed(slope)}"


def waiting(sweep: Sweep, part: str) -> str:
    number = sweep.waiting_at[part]
    return short(number) if part == "loss" else signed(number)


def sweep_lines(links: list[Link], sweep: Sweep) -> list[str]:
    lines = [f"{'loss':8}{waiting(sweep, 'loss')}"]
    for link in links:
        lines.append(f"{link.bottom:8}{waiting(sweep, link.top)} x {short(link.ratio)} "
                     f"= {waiting(sweep, link.bottom)}")
    return lines


def own_link_line(own: Link, sweep: Sweep) -> str:
    slope = sweep.slopes[own.bottom]
    return f"{own.bottom}  {short(own.ratio)} x {waiting(sweep, own.top)} = {signed(slope)}"


def three_ways_lines(nudged: dict[str, float], linked: dict[str, float],
                     swept: dict[str, float]) -> list[str]:
    lines = ["      nudged  links  sweep"]
    for name in swept:
        lines.append(f"{name}    {signed(nudged[name]):8}{signed(linked[name]):7}{signed(swept[name])}")
    return lines


def step_lines(learning_rate: float, name: str, old: float, slope: float, new: float,
               before: ForwardPass, after: ForwardPass) -> list[str]:
    return [
        f"learning rate {short(learning_rate)}: new = old - {short(learning_rate)} x slope, all 13 at once",
        f"{name}: {short(old)} - {short(learning_rate)} x {signed(slope)} = {short(new)}",
        f"hidden layer: h1 {short(after.h1)}, h2 {short(after.h2)}, h3 {short(after.h3)}",
        f"answer y {before.y:.2f} -> {after.y:.2f}",
        f"loss {loss_text(before.loss)} -> {loss_text(after.loss)}",
    ]


def rounds_lines(run: TrainingRun, rounds_to_show: tuple[int, ...]) -> list[str]:
    lines = [f"round {r:2}  loss {loss_text(run.losses[r])}" for r in rounds_to_show]
    end = run.answers[-1]
    lines.append(f"after {len(run.passes) - 1} rounds: answer y {end:.2f}, "
                 f"loss {loss_text(run.losses[-1])}, {end:.0%} sure the mango is ripe")
    return lines
