"""The same forward pass with NumPy: each layer is one matrix multiplication."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .network import ForwardPass, Layer, Network
from .neuron import Activation


@dataclass(frozen=True, eq=False)
class MatrixLayer:
    W: np.ndarray
    b: np.ndarray
    activation: Activation

    @classmethod
    def from_layer(cls, layer: Layer) -> "MatrixLayer":
        W = np.array([neuron.weights for neuron in layer.neurons])  # (neurons, inputs)
        b = np.array([neuron.bias for neuron in layer.neurons])     # (neurons,)
        return cls(W, b, layer.activation)

    def z(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b  # (neurons, inputs) @ (inputs,) + (neurons,) -> (neurons,)

    def fire(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.activation(z) for z in self.z(x)])


@dataclass(frozen=True, eq=False)
class MatrixNetwork:
    layers: tuple[MatrixLayer, ...]

    @classmethod
    def from_network(cls, network: Network) -> "MatrixNetwork":
        return cls(tuple(MatrixLayer.from_layer(layer) for layer in network.layers))

    def forward(self, inputs: Sequence[float]) -> list[np.ndarray]:
        x = np.array(inputs)
        values = []
        for layer in self.layers:
            x = layer.fire(x)
            values.append(x)
        return values


def same_values(by_neuron: ForwardPass, by_matrix: list[np.ndarray]) -> bool:
    if len(by_matrix) != len(by_neuron.layers):
        return False
    return all(np.allclose(by_neuron.values(i), layer) for i, layer in enumerate(by_matrix))
