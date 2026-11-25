"""Optimizers: rules that turn a weight's slope into its next step."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Optimizer(ABC):
    """Moves one weight a step downhill. It may remember that weight's past slopes."""

    @abstractmethod
    def step(self, weight: float, slope: float) -> float: ...

    @abstractmethod
    def fresh(self) -> "Optimizer":
        """A new optimizer with the same settings and an empty memory."""


@dataclass
class NeuronOptimizers:
    """One optimizer for each of a neuron's weights, and one for its bias."""

    weights: list[Optimizer]
    bias: Optimizer
