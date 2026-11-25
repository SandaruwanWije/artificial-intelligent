"""Gradient clipping: a limit on how long all the slopes together may be."""

import math

from neural_network.backpropagation import NeuronSlopes


class GradientClipping:
    """Measures the length of all the slopes together. If it is over the limit, shrinks every slope the same way."""

    def __init__(self, limit: float):
        self.limit = limit

    def length(self, slopes: list[list[NeuronSlopes]]) -> float:
        """The root of every slope squared and added up: how long the whole gradient is."""
        total = 0.0
        for layer_number in range(len(slopes)):
            for n in range(len(slopes[layer_number])):
                neuron_slopes = slopes[layer_number][n]
                for i in range(len(neuron_slopes.weights)):
                    total = total + neuron_slopes.weights[i] * neuron_slopes.weights[i]
                total = total + neuron_slopes.bias * neuron_slopes.bias
        return math.sqrt(total)

    def clipped(self, slopes: list[list[NeuronSlopes]]) -> list[list[NeuronSlopes]]:
        """The same slopes when they are short enough. Otherwise all of them, shrunk to the limit."""
        length = self.length(slopes)
        if length <= self.limit:
            return slopes
        # Every slope is multiplied by the same number, so the direction stays and only the size changes.
        shrink = self.limit / length
        all_clipped = []
        for layer_number in range(len(slopes)):
            layer_clipped = []
            for n in range(len(slopes[layer_number])):
                neuron_slopes = slopes[layer_number][n]
                weights = []
                for i in range(len(neuron_slopes.weights)):
                    weights.append(neuron_slopes.weights[i] * shrink)
                layer_clipped.append(NeuronSlopes(weights, neuron_slopes.bias * shrink))
            all_clipped.append(layer_clipped)
        return all_clipped


NO_LIMIT = GradientClipping(math.inf)
