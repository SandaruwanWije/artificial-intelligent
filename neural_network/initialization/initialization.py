"""Initialization: random starting weights, sized by how many inputs a neuron has."""

import math
from abc import ABC, abstractmethod
from random import Random

from neural_network.activations import Activation
from neural_network.network import Layer
from neural_network.neuron import Neuron


class Initialization(ABC):
    """Builds layers of new neurons. Every weight starts random, so no two neurons start the same."""

    def __init__(self, random: Random):
        self.random = random

    @abstractmethod
    def spread(self, inputs: int) -> float:
        """How far from 0 the weights usually lie, for a neuron with this many inputs."""

    def layer(self, inputs: int, neurons: int, activation: Activation) -> Layer:
        """A layer of new neurons. Each weight is drawn around 0 with the rule's spread. Each bias is 0."""
        spread = self.spread(inputs)
        made = []
        for n in range(neurons):
            weights = []
            for i in range(inputs):
                weights.append(self.random.gauss(0.0, spread))
            made.append(Neuron(weights, 0.0, activation))
        return Layer(made)


class Xavier(Initialization):
    """A spread of 1 over the square root of the inputs. For tanh and sigmoid."""

    def spread(self, inputs: int) -> float:
        return 1 / math.sqrt(inputs)


class He(Initialization):
    """A spread of the square root of 2 over the square root of the inputs. For ReLU."""

    def spread(self, inputs: int) -> float:
        return math.sqrt(2) / math.sqrt(inputs)
