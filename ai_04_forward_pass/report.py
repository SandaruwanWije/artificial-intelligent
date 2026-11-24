"""Turning results into the lines the program prints. Nothing here does maths."""

from collections.abc import Sequence

import numpy as np

from .answer import Gap
from .matrix import MatrixLayer
from .network import ForwardPass, Network
from .neuron import Working
from .work import Chip, Step


def number(value: float) -> str:
    return f"{value:g}"


def rounded(value: float) -> str:
    return f"{round(value, 2):g}"


def numbers(values: Sequence[float]) -> str:
    return ", ".join(number(v) for v in values)


def network_lines(network: Network, run: ForwardPass) -> list[str]:
    x1, x2 = run.inputs
    return [
        f"a mango: x1 = {number(x1)} (how yellow), x2 = {number(x2)} (how soft)",
        f"{network.weight_count} weights + {network.bias_count} biases"
        f" = {network.parameter_count} parameters",
        f"hidden layer: {numbers(run.values(0))}",
        f"output: {run.answer:.2f}, so {run.answer:.0%} sure the mango is ripe",
    ]


def neuron_lines(working: Working) -> list[str]:
    neuron = working.neuron
    multiply = " + ".join(f"{number(w)} x {number(x)}" for w, x in zip(neuron.weights, working.inputs))
    add = " + ".join(number(p) for p in (*working.products, neuron.bias))
    off = "  (switched off)" if working.switched_off else ""
    return [
        f"{neuron.name}: {multiply} + {number(neuron.bias)}",
        f"{' ' * len(neuron.name)}  = {add} = {number(working.z)}"
        f"   {neuron.activation.name} -> {rounded(working.value)}{off}",
    ]


def layer_lines(layer_name: str, matrix_name: str, names: Sequence[str], layer: MatrixLayer,
                x: np.ndarray, values: np.ndarray, formula: str) -> list[str]:
    rows, columns = layer.W.shape
    lines = [f"{layer_name} layer, {matrix_name} is {rows} x {columns}: one row per neuron"]
    for name, row, bias, z in zip(names, layer.W, layer.b, layer.z(x)):
        weights = " ".join(f"{w:5g}" for w in row)
        lines.append(f"  {name:3} [{weights}] . [{numbers(x)}] + {number(bias):4}  = {number(z)}")
    lines.append(f"  {formula} = [{', '.join(rounded(v) for v in values)}]")
    return lines


def work_lines(per_layer: Sequence[int], chips: dict[Chip, list[Step]], digit_reader: int) -> list[str]:
    hidden, output = per_layer
    lines = [f"multiplications: {hidden} in the hidden layer + {output} in the output"
             f" = {sum(per_layer)}"]
    for chip, steps in chips.items():
        at_a_time = "one at a time" if chip.cores == 1 else "a whole layer at once"
        lines.append(f"{chip.name}: {len(steps)} steps, {at_a_time}")
    lines.append(f"the digit reader in article 21: {digit_reader:,} multiplications per picture")
    return lines


def gap_lines(gap: Gap) -> list[str]:
    return [
        f"the network says {gap.guess:.2f}: {gap.guess:.0%} sure the mango is ripe",
        f"the right answer is {number(gap.right_answer)}, so it is {gap.short_by:.2f} short",
    ]
