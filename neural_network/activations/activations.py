"""Activations: the bend at the end of a neuron, and how steep the bend is (its slope)."""

import math
from abc import ABC, abstractmethod


class Activation(ABC):
    """Turns a neuron's total into its output, and says how steep it is there."""

    @abstractmethod
    def value(self, z: float) -> float: ...

    @abstractmethod
    def slope(self, z: float) -> float: ...

    def __call__(self, z: float) -> float:
        return self.value(z)


class Step(Activation):
    """1 above 0, else 0. The perceptron's hard edge."""

    def value(self, z: float) -> float:
        if z > 0:
            return 1.0
        return 0.0

    def slope(self, z: float) -> float:
        return 0.0


class Sigmoid(Activation):
    """A smooth step from 0 to 1."""

    def value(self, z: float) -> float:
        return 1 / (1 + math.exp(-z))

    def slope(self, z: float) -> float:
        s = self.value(z)
        return s * (1 - s)


class Tanh(Activation):
    """A smooth step from -1 to 1."""

    def value(self, z: float) -> float:
        up = math.exp(z)
        down = math.exp(-z)
        return (up - down) / (up + down)

    def slope(self, z: float) -> float:
        t = self.value(z)
        return 1 - t * t


class ReLU(Activation):
    """0 for anything below 0; the total itself above 0."""

    def value(self, z: float) -> float:
        if z > 0:
            return z
        return 0.0

    def slope(self, z: float) -> float:
        if z > 0:
            return 1.0
        return 0.0


class Gelu(Activation):
    """A smooth ReLU that dips a little below 0. This is the tanh form."""

    SCALE = math.sqrt(2 / math.pi)
    CUBE = 0.044715

    def value(self, z: float) -> float:
        inner = self.SCALE * (z + self.CUBE * z * z * z)
        return 0.5 * z * (1 + math.tanh(inner))

    def slope(self, z: float) -> float:
        inner = self.SCALE * (z + self.CUBE * z * z * z)
        t = math.tanh(inner)
        inner_slope = self.SCALE * (1 + 3 * self.CUBE * z * z)
        return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * inner_slope


class NoBend(Activation):
    """Passes the total straight through."""

    def value(self, z: float) -> float:
        return z

    def slope(self, z: float) -> float:
        return 1.0


STEP, SIGMOID, TANH, RELU, GELU, NO_BEND = Step(), Sigmoid(), Tanh(), ReLU(), Gelu(), NoBend()
