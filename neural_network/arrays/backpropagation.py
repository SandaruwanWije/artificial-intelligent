"""Backpropagation for arrays: the slope of every weight and bias, for all the examples in one sweep backwards."""

from dataclasses import dataclass

import numpy as np

from neural_network.arrays.forward_pass import ArrayForwardPass
from neural_network.arrays.loss import ArrayLoss
from neural_network.arrays.network import ArrayNetwork


@dataclass
class LayerSlopes:
    """How the loss moves when each weight or bias of one layer moves a little. Each array has its layer's shape."""

    weights: np.ndarray
    biases: np.ndarray


class ArrayBackpropagation:
    """Starts at the loss and walks back through the layers, with the blame of every example in one array."""

    def __init__(self, loss: ArrayLoss):
        self.loss = loss

    def slopes(self, network: ArrayNetwork, inputs: np.ndarray, truths: np.ndarray) -> list[LayerSlopes]:
        """One LayerSlopes per layer, in the same order as the network. Each slope is the average over the examples."""
        forward = ArrayForwardPass(network, inputs)
        examples = inputs.shape[0]
        # The blame starts at the answers. Dividing by the number of examples makes every slope an average.
        blame = self.loss.slopes(forward.answer, truths) / examples
        all_slopes = []
        # Then it walks back, from the last layer to the first.
        layer_number = len(network.layers) - 1
        while layer_number >= 0:
            layer = network.layers[layer_number]
            step = forward.steps[layer_number]
            # Back through the bend: the blame at every total.
            at_totals = blame * layer.activation.slope(step.totals)
            # A weight's slope is the blame at the total times the input it multiplied, added up over the examples.
            weight_slopes = step.inputs.T @ at_totals
            # The bias is added straight to the total, so its slope is the blame at the total, added up.
            bias_slopes = at_totals.sum(axis=0)
            all_slopes.insert(0, LayerSlopes(weight_slopes, bias_slopes))
            # Each input gets blame back along every weight it went through.
            blame = at_totals @ layer.weights.T
            layer_number = layer_number - 1
        return all_slopes
