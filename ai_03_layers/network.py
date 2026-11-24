"""Layers of neurons, and a network that hands each layer's answers to the next."""

from collections.abc import Sequence
from dataclasses import dataclass

from .neuron import Neuron


@dataclass(frozen=True)
class Layer:
    neurons: tuple[Neuron, ...]

    def answer(self, inputs: Sequence[float]) -> tuple[float, ...]:
        return tuple(neuron.answer(inputs) for neuron in self.neurons)


@dataclass(frozen=True)
class Network:
    layers: tuple[Layer, ...]

    def answers_layer_by_layer(self, inputs: Sequence[float]) -> list[tuple[float, ...]]:
        answers = [tuple(inputs)]
        for layer in self.layers:
            answers.append(layer.answer(answers[-1]))
        return answers

    def new_numbers(self, inputs: Sequence[float]) -> tuple[float, ...]:
        return self.answers_layer_by_layer(inputs)[1]

    def answer(self, inputs: Sequence[float]) -> float:
        return self.answers_layer_by_layer(inputs)[-1][0]

    @property
    def hidden_neurons(self) -> tuple[Neuron, ...]:
        return tuple(neuron for layer in self.layers[:-1] for neuron in layer.neurons)

    @property
    def output_neuron(self) -> Neuron:
        return self.layers[-1].neurons[0]
