"""The neuron: weights, a bias and an activation."""

from neural_network.activations import RELU, Activation


class Neuron:
    """Multiplies each input by its weight, adds them up with the bias, then bends the total."""

    def __init__(self, weights: list[float], bias: float, activation: Activation = RELU):
        self.weights = list(weights)  # a copy, so learning never changes the caller's list
        self.bias = bias
        self.activation = activation

    def total(self, inputs: list[float]) -> float:
        total = 0.0
        for i in range(len(self.weights)):
            total = total + self.weights[i] * inputs[i]
        total = total + self.bias
        return total

    def output(self, inputs: list[float]) -> float:
        total = self.total(inputs)
        return self.activation(total)
