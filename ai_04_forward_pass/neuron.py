"""One neuron and its three moves: multiply, add, activate."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Activation:
    name: str
    rule: Callable[[float], float]

    def __call__(self, z: float) -> float:
        return self.rule(z)


RELU = Activation("ReLU", lambda z: max(0.0, z))
SIGMOID = Activation("sigmoid", lambda z: 1 / (1 + math.exp(-z)))


@dataclass(frozen=True)
class Neuron:
    name: str
    weights: tuple[float, ...]
    bias: float
    activation: Activation

    def fire(self, inputs: Sequence[float]) -> "Working":
        products = tuple(weight * x for weight, x in zip(self.weights, inputs))
        z = sum(products) + self.bias
        value = self.activation(z)
        return Working(self, tuple(inputs), products, z, value)


@dataclass(frozen=True)
class Working:
    """One neuron's sum, written out the way you would on paper."""

    neuron: Neuron
    inputs: tuple[float, ...]
    products: tuple[float, ...]
    z: float
    value: float

    @property
    def switched_off(self) -> bool:
        return self.value == 0
