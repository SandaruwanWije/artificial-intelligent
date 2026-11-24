"""The lines the program prints. Nothing here does maths."""

from .cloud import Point
from .neuron import Neuron, hard_edge, no_bend, relu, sigmoid
from .tents import DeepOrWide
from .trainer import Trial
from .xor import BAND, Corner

ACTIVATION_NAMES = {relu: "ReLU", sigmoid: "sigmoid", no_bend: "no bend", hard_edge: "hard edge"}


def number(value: float) -> str:
    return f"{round(float(value), 2) + 0.0:g}"


def pair(values: tuple[float, ...]) -> str:
    return f"({', '.join(number(v) for v in values)})"


def neuron_line(name: str, neuron: Neuron) -> str:
    weights = ", ".join(f"w{i} {number(w)}" for i, w in enumerate(neuron.weights, start=1))
    return f"{name}: {weights}, bias {number(neuron.bias)}, {ACTIVATION_NAMES[neuron.activation]}"


def try_line(neuron: Neuron, wrong: list[Corner]) -> str:
    w1, w2 = neuron.weights
    missed = ", ".join(pair(corner.inputs) for corner in wrong)
    return (f"w1 {number(w1)}, w2 {number(w2)}, bias {number(neuron.bias)}: "
            f"{4 - len(wrong)} of 4 right, wrong at {missed}")


def band_line(corner: Corner, answer: int) -> str:
    low, high = BAND
    total = corner.x1 + corner.x2
    where = f"below {low}" if total < low else f"above {high}" if total > high else "between the lines"
    return f"{pair(corner.inputs)}  x1 + x2 = {total}  {where}, so {answer}"


def corner_line(corner: Corner, new_numbers: tuple[float, ...], y: float) -> str:
    h1, h2 = new_numbers
    return f"{pair(corner.inputs)}  h1 = {number(h1)}  h2 = {number(h2)}  y = {number(y)}"


def trial_line(trial: Trial, points: tuple[Point, ...]) -> str:
    n = trial.hidden_neurons
    lines = len(trial.lines_switched_on())
    return (f"{n} hidden neuron{'s' if n > 1 else ''}, start {trial.seed}: "
            f"{trial.percent_right(points)}% right, {len(trial.wrong_points(points))} wrong, "
            f"{lines} line{'s' if lines > 1 else ''}")


def opened_up(trial: Trial) -> list[str]:
    hidden = [neuron_line(f"h{j}", n) for j, n in enumerate(trial.network.hidden_neurons, 1)]
    return hidden + [neuron_line("y", trial.network.output_neuron)]


def depth_table(rows: list[DeepOrWide]) -> list[str]:
    lines = ["layers  deep neurons  corners  wide neurons"]
    for row in rows:
        lines.append(f"{row.layers:>6}  {row.deep_neurons:>12}  {row.corners:>7}  {row.wide_neurons:>12}")
    return lines
