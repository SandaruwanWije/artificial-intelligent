"""Turning results into the lines the program prints; nothing here does maths."""

from .bowl import Gradient, Weights
from .descent import BowlDescent, Step, ValleyDescent


def weight_line(weight: float, loss: float) -> str:
    return f"weight {weight:.2f}  loss {loss:.2f}"


def slope_line(weight: float, slope: float, felt: float) -> str:
    return f"weight {weight:.2f}  slope {slope:.2f}  felt by a nudge {felt:.2f}"


def uphill_line(learning_rate: float, step: Step) -> str:
    return (f"{learning_rate} x {step.slope:.2f} = {-step.size:.2f} points uphill, "
            f"so the step is {step.size:+.2f}")


def step_line(step: Step) -> str:
    return f"weight {step.weight:5.2f}   slope {step.slope:5.2f}   step {step.size:+.2f}"


def step_sizes_line(descent: ValleyDescent) -> str:
    return "step sizes: " + ", ".join(f"{step.size:.2f}" for step in descent.steps)


def weights_line(weights: Weights, loss: float) -> str:
    return f"w1 {weights.w1:.3g}, w2 {weights.w2:.3g}, loss {round(loss, 3):g}"


def gradient_lines(gradient: Gradient, felt: Gradient) -> list[str]:
    return [
        f"slope for w1 alone {gradient.slope_for_w1:g}, for w2 alone {gradient.slope_for_w2:g}",
        f"felt by nudges: {felt.slope_for_w1:.1f}, {felt.slope_for_w2:.1f}",
        f"gradient ({gradient.slope_for_w1:g}, {gradient.slope_for_w2:g})",
    ]


def step_against_line(learning_rate: float, start: Weights, after: Weights) -> str:
    return (f"step = -{learning_rate} x gradient = "
            f"({after.w1 - start.w1:.2f}, {after.w2 - start.w2:.2f})")


def loss_table(descents: dict[str, BowlDescent], every: int) -> list[str]:
    last = len(next(iter(descents.values())).losses) - 1
    shown = range(0, last + 1, every)
    lines = ["learning rate      step" + "".join(f"{step:>7}" for step in shown)]
    for name, descent in descents.items():
        label = f"{descent.learning_rate} {name}"
        losses = "".join(f"{round(descent.losses[step], 3):>7g}" for step in shown)
        lines.append(f"{label:<22}{losses}")
    return lines


def climb_line(name: str, descent: BowlDescent) -> str:
    step = descent.lowest_loss_step
    return (f"{descent.learning_rate} {name}: falls to {round(descent.losses[step], 3):g} "
            f"at step {step}, then climbs to {round(descent.losses[-1], 3):g}")
