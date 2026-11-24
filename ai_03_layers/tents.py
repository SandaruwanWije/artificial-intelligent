"""The tent: two ReLU neurons that fold 0 to 1 onto itself, stacked layer after layer."""

from dataclasses import dataclass

import numpy as np

from .network import Layer, Network
from .neuron import Neuron, no_bend, relu

A = Neuron(weights=(1,), bias=0, activation=relu)
B = Neuron(weights=(1,), bias=-0.5, activation=relu)
TENT = Neuron(weights=(2, -4), bias=0, activation=no_bend)
NEURONS_PER_LAYER = 2


def deep_tents(layers: int) -> Network:
    return Network((Layer((A, B)), Layer((TENT,))) * layers)


def corners(network: Network, samples: int = 2**14) -> int:
    x = np.linspace(0, 1, samples + 1)
    slopes = np.diff(network.answer((x,))) * samples
    return int(np.count_nonzero(np.diff(slopes)))


@dataclass(frozen=True)
class DeepOrWide:
    layers: int
    deep_neurons: int
    corners: int

    @property
    def wide_neurons(self) -> int:
        return self.corners  # each ReLU neuron in one layer bends the line in one place

    @classmethod
    def for_layers(cls, layers: int) -> "DeepOrWide":
        return cls(layers, NEURONS_PER_LAYER * layers, corners(deep_tents(layers)))
