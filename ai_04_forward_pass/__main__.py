"""The forward pass article as a program, one function per chapter, then the pictures."""

from pathlib import Path

import numpy as np

from .answer import Gap
from .diagrams import draw_matrix, draw_network
from .figures import draw_gap, draw_sums, draw_work
from .mango import MANGO, NETWORK, RIPE
from .matrix import MatrixNetwork, same_values
from .network import ForwardPass
from .race import race
from .report import gap_lines, layer_lines, network_lines, neuron_lines, work_lines
from .work import CPU_CORE, DIGIT_READER, GPU, Chip, Step, multiplications_per_layer

IMAGES = Path(__file__).parent / "images"
MATRIX_NETWORK = MatrixNetwork.from_network(NETWORK)
FORMULAS = ("h = ReLU(W1 . x + b1)", "y = sigmoid(W2 . h + b2)")


def numbers_in_answer_out() -> ForwardPass:
    print("Numbers in, answer out")
    run = NETWORK.forward(MANGO)
    for line in network_lines(NETWORK, run):
        print(f"  {line}")
    return run


def one_neuron_at_a_time(run: ForwardPass) -> None:
    print("\nOne neuron at a time")
    for working in run.workings:
        for line in neuron_lines(working):
            print(f"  {line}")


def a_layer_in_one_move(run: ForwardPass) -> list[np.ndarray]:
    print("\nA layer in one move")
    by_matrix = MATRIX_NETWORK.forward(MANGO)
    layer_inputs = [np.array(MANGO), *by_matrix[:-1]]
    layers = zip(NETWORK.layers, MATRIX_NETWORK.layers, layer_inputs, by_matrix, FORMULAS)
    for number, (layer, matrix_layer, x, values, formula) in enumerate(layers, start=1):
        names = [neuron.name for neuron in layer.neurons]
        for line in layer_lines(layer.name, f"W{number}", names, matrix_layer, x, values, formula):
            print(f"  {line}")
    if same_values(run, by_matrix):
        print("  every value matches the pass neuron by neuron")
    return by_matrix


def why_graphics_cards() -> dict[Chip, list[Step]]:
    print("\nWhy graphics cards")
    per_layer = multiplications_per_layer(NETWORK.layer_sizes)
    schedules = {chip: chip.schedule(per_layer) for chip in (CPU_CORE, GPU)}
    digit_reader = sum(multiplications_per_layer(DIGIT_READER))
    for line in work_lines(per_layer, schedules, digit_reader):
        print(f"  {line}")
    timed = race()
    print(f"  timed here, one layer of the digit reader: neuron by neuron"
          f" {timed.neuron_by_neuron_ms:.1f} ms, one matrix {timed.one_matrix_ms:.2f} ms"
          f" (changes every run)")
    return schedules


def how_wrong_is_it(run: ForwardPass) -> Gap:
    print("\nHow wrong is it?")
    gap = Gap(guess=run.answer, right_answer=RIPE)
    for line in gap_lines(gap):
        print(f"  {line}")
    return gap


def main() -> None:
    run = numbers_in_answer_out()
    one_neuron_at_a_time(run)
    by_matrix = a_layer_in_one_move(run)
    schedules = why_graphics_cards()
    gap = how_wrong_is_it(run)

    draw_network(run, IMAGES / "network.png")
    draw_sums(run, IMAGES / "sums.png")
    draw_matrix(NETWORK, MATRIX_NETWORK, by_matrix, MANGO, IMAGES / "matrix.png")
    draw_work(schedules, IMAGES / "work.png")
    draw_gap(gap, IMAGES / "gap.png")
    print(f"\nPictures saved to {IMAGES}")


if __name__ == "__main__":
    main()
