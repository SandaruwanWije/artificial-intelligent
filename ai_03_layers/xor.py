"""XOR's four corners, the lines one neuron tries, and the network that solves it."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from .network import Layer, Network
from .neuron import Neuron, hard_edge, no_bend, relu


@dataclass(frozen=True)
class Corner:
    x1: int
    x2: int
    wants: int

    @property
    def inputs(self) -> tuple[int, int]:
        return (self.x1, self.x2)


CORNERS = (Corner(0, 0, 0), Corner(0, 1, 1), Corner(1, 0, 1), Corner(1, 1, 0))

LINES_ONE_NEURON_TRIES = tuple(
    Neuron((w1, w2), bias, hard_edge)
    for w1, w2, bias in ((0, 1, -0.5), (1, 1, -0.5), (-1, 1, -0.5), (1, -1, -0.5), (-1, -1, 1.5))
)


def wrong_corners(neuron: Neuron) -> list[Corner]:
    return [corner for corner in CORNERS if neuron.answer(corner.inputs) != corner.wants]


def corners_right(neuron: Neuron) -> int:
    return len(CORNERS) - len(wrong_corners(neuron))


def every_line_on_a_grid() -> list[Neuron]:
    weights = np.arange(-2, 2.01, 0.5)
    biases = np.arange(-2, 2.01, 0.25)
    return [Neuron((w1, w2), bias, hard_edge) for w1, w2, bias in product(weights, weights, biases)]


BAND = (0.5, 1.5)


def band_answer(corner: Corner) -> int:
    low, high = BAND
    return 1 if low < corner.x1 + corner.x2 < high else 0


H1 = Neuron(weights=(1, 1), bias=0, activation=relu)
H2 = Neuron(weights=(1, 1), bias=-1, activation=relu)
Y = Neuron(weights=(1, -2), bias=0, activation=no_bend)
XOR_NETWORK = Network(layers=(Layer((H1, H2)), Layer((Y,))))
OUTPUT_LINE = 0.5
