"""Both parts of the backpropagation article as a program: one function per chapter, then the pictures."""

from pathlib import Path

from .backprop import Backprop, Sweep
from .figures import draw_agreement, draw_formulas, draw_nudge, draw_training
from .network import Example, Network, Parameters
from .network_figures import draw_one_at_a_time, draw_sweep
from .nudge import Nudge, NudgeTest
from .report import formula_lines, mango_lines, nudge_lines, parameter_line, road_length_line, road_line
from .report_part2 import (link_lines, own_link_line, product_line, rounds_lines, shortcut_lines,
                           step_lines, sweep_lines, three_ways_lines)
from .trainer import Trainer, TrainingRun

IMAGES = Path(__file__).parent / "images"

MANGO = Example(x1=1.0, x2=0.5, right_answer=1.0)
ARTICLE_7_NETWORK = Network(Parameters(
    w1=0.8, w2=-0.4, b1=0.1,
    w3=0.3, w4=0.9, b2=-0.2,
    w5=-0.6, w6=0.5, b3=0.3,
    w7=1.2, w8=-0.7, w9=0.5, b4=-0.1,
))
NUDGE_TEST = NudgeTest(size=0.01)
TRAINER = Trainer(learning_rate=0.5, rounds=40)
SWEEP_ORDER = ("w7", "w8", "w9", "b4", "w1", "w2", "b1", "w3", "w4", "b2", "w5", "w6", "b3")
ROUNDS_TO_SHOW = (0, 1, 2, 5, 10, 20, 30, 40)


def chapter(title: str, lines: list[str]) -> None:
    print(title)
    for line in lines:
        print(f"  {line}")
    print()


def who_is_to_blame() -> None:
    passed = ARTICLE_7_NETWORK.forward(MANGO)
    chapter("Who is to blame?", [*mango_lines(MANGO, passed), parameter_line(ARTICLE_7_NETWORK.parameters)])


def one_at_a_time() -> dict[str, Nudge]:
    nudges = NUDGE_TEST.nudge_each(ARTICLE_7_NETWORK, MANGO)
    values = ARTICLE_7_NETWORK.parameters
    chapter("One at a time", [road_line(nudge, values.value(name)) for name, nudge in nudges.items()])
    return nudges


def the_same_thing_in_numbers(nudges: dict[str, Nudge], backprop: Backprop) -> None:
    roads = {name: backprop.road(name) for name in nudges}
    lines = nudge_lines(nudges["w1"], ARTICLE_7_NETWORK.parameters.w1)
    chapter("The same thing, in numbers", [*lines, *formula_lines("w1", roads["w1"]), road_length_line(roads)])


def link_by_link(backprop: Backprop) -> dict[str, float]:
    lines = ["w1's road, link by link:", *[f"  {line}" for line in link_lines(backprop.road("w1"))]]
    lines += shortcut_lines(backprop)
    linked = {name: backprop.slope_link_by_link(name) for name in backprop.parameters.names()}
    lines.append("every number, its links multiplied:")
    lines += [f"  {product_line(name, backprop.road(name), slope)}" for name, slope in linked.items()]
    chapter("Link by link", lines)
    return linked


def backwards_in_one_sweep(backprop: Backprop, nudges: dict[str, Nudge],
                           linked: dict[str, float]) -> Sweep:
    sweep = backprop.sweep()
    lines = sweep_lines(backprop.links, sweep)
    lines.append("each number alone: its own link x the number waiting where it feeds")
    lines += [f"  {own_link_line(backprop.own_link(name), sweep)}" for name in SWEEP_ORDER]
    nudged = {name: nudge.slope for name, nudge in nudges.items()}
    lines.append("the same 13 slopes, three ways:")
    lines += [f"  {line}" for line in three_ways_lines(nudged, linked, sweep.slopes)]
    chapter("Backwards, in one sweep", lines)
    return sweep


def one_step(sweep: Sweep) -> None:
    stepped = TRAINER.step(ARTICLE_7_NETWORK, MANGO)
    before, after = ARTICLE_7_NETWORK.forward(MANGO), stepped.forward(MANGO)
    old, new = ARTICLE_7_NETWORK.parameters.w1, stepped.parameters.w1
    chapter("One step", step_lines(TRAINER.learning_rate, "w1", old, sweep.slopes["w1"], new, before, after))


def round_and_round() -> TrainingRun:
    run = TRAINER.train(ARTICLE_7_NETWORK, MANGO)
    chapter("Round and round", rounds_lines(run, ROUNDS_TO_SHOW))
    return run


def main() -> None:
    backprop = Backprop(ARTICLE_7_NETWORK, MANGO)
    passed = ARTICLE_7_NETWORK.forward(MANGO)

    who_is_to_blame()
    nudges = one_at_a_time()
    the_same_thing_in_numbers(nudges, backprop)

    linked = link_by_link(backprop)
    sweep = backwards_in_one_sweep(backprop, nudges, linked)
    one_step(sweep)
    run = round_and_round()

    draw_one_at_a_time(nudges, ARTICLE_7_NETWORK.parameters, passed, IMAGES / "one_at_a_time.png")
    draw_nudge(ARTICLE_7_NETWORK, MANGO, nudges["w1"], IMAGES / "nudge.png")
    draw_formulas(backprop, IMAGES / "formulas.png")
    draw_sweep(sweep, passed, IMAGES / "sweep.png")
    draw_agreement(nudges, sweep, IMAGES / "agreement.png")
    draw_training(run, IMAGES / "training.png")
    print(f"Pictures saved to {IMAGES}")


if __name__ == "__main__":
    main()
