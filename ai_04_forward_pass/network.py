"""A network as layers of neurons, and a forward pass worked out neuron by neuron."""

from collections.abc import Sequence
from dataclasses import dataclass

from .neuron import Activation, Neuron, Working


@dataclass(frozen=True)
class Layer:
    name: str
    neurons: tuple[Neuron, ...]

    @property
    def activation(self) -> Activation:
        return self.neurons[0].activation

    def fire(self, inputs: Sequence[float]) -> tuple[Working, ...]:
        return tuple(neuron.fire(inputs) for neuron in self.neurons)


@dataclass(frozen=True)
class ForwardPass:
    inputs: tuple[float, ...]
    layers: tuple[tuple[Working, ...], ...]

    @property
    def workings(self) -> list[Working]:
        return [working for layer in self.layers for working in layer]

    def values(self, layer: int) -> tuple[float, ...]:
        return tuple(working.value for working in self.layers[layer])

    @property
    def answer(self) -> float:
        return self.values(-1)[0]


@dataclass(frozen=True)
class Network:
    layers: tuple[Layer, ...]

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        first = self.layers[0].neurons[0]
        return (len(first.weights),) + tuple(len(layer.neurons) for layer in self.layers)

    @property
    def weight_count(self) -> int:
        return sum(len(neuron.weights) for layer in self.layers for neuron in layer.neurons)

    @property
    def bias_count(self) -> int:
        return sum(len(layer.neurons) for layer in self.layers)

    @property
    def parameter_count(self) -> int:
        return self.weight_count + self.bias_count

    def forward(self, inputs: Sequence[float]) -> ForwardPass:
        values = tuple(inputs)
        layers = []
        for layer in self.layers:
            workings = layer.fire(values)
            layers.append(workings)
            values = tuple(working.value for working in workings)
        return ForwardPass(tuple(inputs), tuple(layers))
