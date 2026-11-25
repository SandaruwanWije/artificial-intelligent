"""Layers of neurons, and a network that passes each layer's outputs to the next layer."""

from neural_network.neuron import Neuron


class Layer:
    """Neurons side by side. Every neuron gets the same inputs."""

    def __init__(self, neurons: list[Neuron]):
        self.neurons = list(neurons)

    def outputs(self, inputs: list[float]) -> list[float]:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output(inputs))
        return outputs


class Network:
    """Layers in a row. Each layer's outputs are the next layer's inputs."""

    def __init__(self, layers: list[Layer]):
        self.layers = list(layers)

    def answer(self, inputs: list[float]) -> list[float]:
        values = inputs
        for layer in self.layers:
            values = layer.outputs(values)
        return values
