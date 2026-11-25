"""Normalization: a layer that gives its numbers a mean of 0 and a spread of 1, then a learned scale and shift."""

import math
from abc import ABC, abstractmethod

from neural_network.activations import Activation
from neural_network.network import Layer
from neural_network.neuron import Neuron


class Normalization(Layer, ABC):
    """A layer whose neurons each scale, shift and bend one standardized number. Each kind picks the group."""

    TINY = 1e-8  # keeps the division safe when all the numbers of a group are the same

    def __init__(self, size: int, activation: Activation):
        neurons = []
        for n in range(size):
            # Each neuron has one weight, its scale, which starts at 1. Its bias is its shift, which starts at 0.
            neurons.append(Neuron([1.0], 0.0, activation))
        super().__init__(neurons)

    def mean(self, numbers: list[float]) -> float:
        total = 0.0
        for i in range(len(numbers)):
            total = total + numbers[i]
        return total / len(numbers)

    def spread(self, numbers: list[float]) -> float:
        """A typical distance from the mean: the root of the average squared distance."""
        mean = self.mean(numbers)
        squares = 0.0
        for i in range(len(numbers)):
            distance = numbers[i] - mean
            squares = squares + distance * distance
        return math.sqrt(squares / len(numbers) + self.TINY)

    def standardized(self, numbers: list[float]) -> list[float]:
        """Every number minus the mean, divided by the spread. Now their mean is 0 and their spread is 1."""
        mean = self.mean(numbers)
        spread = self.spread(numbers)
        standardized = []
        for i in range(len(numbers)):
            standardized.append((numbers[i] - mean) / spread)
        return standardized

    @abstractmethod
    def inputs_of(self, n: int, inputs: list[float]) -> list[float]:
        """Neuron number n reads one number: its own input, standardized over the kind's group."""

    @abstractmethod
    def blame_on_inputs(self, inputs: list[float], blame_at_totals: list[float]) -> list[float]:
        """Takes the blame back through each neuron's scale, then back through the standardizing."""
