"""A network held as arrays: one grid of weights and one row of biases for each layer."""

import numpy as np

from neural_network.arrays.activations import ArrayActivation
from neural_network.network import Layer


class ExampleArrays:
    """Examples stacked as two arrays: one row of inputs and one row of truths for each example."""

    def __init__(self, examples: list[tuple[list[float], list[float]]]):
        inputs = []
        truths = []
        for example_inputs, example_truths in examples:
            inputs.append(example_inputs)
            truths.append(example_truths)
        self.inputs = np.array(inputs)
        self.truths = np.array(truths)


class ArrayLayer:
    """A layer of neurons as two arrays: weights with one column for each neuron, and one bias for each neuron."""

    def __init__(self, layer: Layer, activation: ArrayActivation):
        rows = []
        biases = []
        for neuron in layer.neurons:
            rows.append(neuron.weights)
            biases.append(neuron.bias)
        # Here each neuron's weights are one row. Turned on its side, each neuron is one column.
        self.weights = np.array(rows).T
        self.biases = np.array(biases)
        self.activation = activation

    def totals(self, inputs: np.ndarray) -> np.ndarray:
        """Every neuron's total for every example: one row for each example, one column for each neuron."""
        return inputs @ self.weights + self.biases

    def outputs(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation.value(self.totals(inputs))


class ArrayNetwork:
    """Layers in a row. Each layer's outputs are the next layer's inputs."""

    def __init__(self, layers: list[ArrayLayer]):
        self.layers = list(layers)

    def answer(self, inputs: np.ndarray) -> np.ndarray:
        values = inputs
        for layer in self.layers:
            values = layer.outputs(values)
        return values
