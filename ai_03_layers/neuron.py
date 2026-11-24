"""One neuron: a weighted sum of its inputs plus a bias, bent by an activation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Activation = Callable[[float], float]


# NumPy versions, so one neuron can answer for a single point or a whole grid of points
def relu(z: float) -> float:
    return np.maximum(0.0, z)


def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def no_bend(z: float) -> float:
    return z


def hard_edge(z: float) -> float:
    return (z > 0) * 1.0


@dataclass(frozen=True)
class Neuron:
    weights: tuple[float, ...]
    bias: float
    activation: Activation = relu

    def weighted_sum(self, inputs: Sequence[float]) -> float:
        return sum(weight * x for weight, x in zip(self.weights, inputs)) + self.bias

    def answer(self, inputs: Sequence[float]) -> float:
        return self.activation(self.weighted_sum(inputs))

    def is_switched_off(self, spots: Sequence[Sequence[float]]) -> bool:
        return all(self.weighted_sum(spot) <= 0 for spot in spots)
