"""The forward pass for arrays: every example goes through a layer in one step, and every step is kept."""

from dataclasses import dataclass

import numpy as np

from neural_network.arrays.network import ArrayNetwork


@dataclass
class ArrayStep:
    """What one layer got, what its neurons added up to, and what came out, with one row for each example."""

    inputs: np.ndarray
    totals: np.ndarray
    outputs: np.ndarray


class ArrayForwardPass:
    """Runs all the examples through a network together and keeps each layer's step."""

    def __init__(self, network: ArrayNetwork, inputs: np.ndarray):
        self.steps: list[ArrayStep] = []
        values = inputs
        for layer in network.layers:
            totals = layer.totals(values)
            outputs = layer.activation.value(totals)
            self.steps.append(ArrayStep(values, totals, outputs))
            values = outputs
        self.answer = values
