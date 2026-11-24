"""Turning results into the lines the program prints."""

import math
from collections.abc import Sequence

from .activations import Activation
from .bump import FinalNeuron
from .layers import Layer
from .trainer import TrainingRun


def num(value: float, places: int) -> str:
    return f"{round(value, places) + 0.0:g}"


def point_line(curve: Activation, z: float) -> str:
    slope = curve.slope(z)
    if math.isnan(slope):
        return f"z = {z:g}  the jump: no slope at all"
    return f"z = {z:g}  output {num(curve.value(z), 2)}  slope {num(slope, 3)}"


def flatness(curve: Activation, far: float) -> str:
    left, right = curve.is_almost_flat(-far), curve.is_almost_flat(far)
    if left and right:
        return "flat at both ends"
    return "flat only on the left" if left else "flat only on the right" if right else "never flat"


def slopes_line(curve: Activation, zs: Sequence[float]) -> str:
    slopes = ", ".join(f"{curve.slope(z):.2g}" for z in zs)
    at = ", ".join(f"{z:g}" for z in zs)
    return f"{curve.name:<8} slope at z = {at}: {slopes:<15} {flatness(curve, max(zs))}"


def lowest_line(curve: Activation, zs: Sequence[float]) -> str:
    z, lowest = curve.lowest(zs)
    return f"{curve.name} dips lowest to {num(lowest, 2)}, near z = {num(z, 2)}"


def layer_line(label: str, layer: Layer, x: float) -> str:
    return f"{label:<10} {str(layer):<14} {x:g} gives {layer(x):g}"


def row(label: str, values: Sequence[float]) -> str:
    return f"{label:<8}" + "".join(f"{v:>4g}" for v in values)


def final_neuron_line(final: FinalNeuron) -> str:
    weights = ", ".join(f"{w:g}" for w in final.weights)
    corners = ", ".join(f"{n.corner:g}" for n in final.inputs)
    return f"corners at x = {corners}; the final neuron weighs them {weights}, bias {final.bias:g}"


def percent_table(runs: Sequence[TrainingRun], steps: Sequence[int]) -> list[str]:
    lines = [f"{'steps':<8}" + "".join(f"{s:>5}" for s in steps)]
    for run in runs:
        percents = "".join(f"{run.percent_right(s):>5}" for s in steps)
        lines.append(f"{run.network.activation.name:<8}" + percents)
    return lines
