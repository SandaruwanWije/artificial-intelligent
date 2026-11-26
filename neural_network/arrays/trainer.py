"""Training for arrays: forward, loss, backward and one step, with all the examples in every step."""

import numpy as np

from neural_network.arrays.backpropagation import ArrayBackpropagation
from neural_network.arrays.network import ArrayNetwork
from neural_network.gradient_descent import GradientDescent


class ArrayTrainer:
    """Trains an array network: finds every slope in one sweep, then moves each whole array one step downhill."""

    def __init__(self, network: ArrayNetwork, backpropagation: ArrayBackpropagation,
                 gradient_descent: GradientDescent):
        self.network = network
        self.backpropagation = backpropagation
        self.gradient_descent = gradient_descent

    def step(self, inputs: np.ndarray, truths: np.ndarray) -> None:
        """One step for every weight and bias, from its average slope over the examples."""
        slopes = self.backpropagation.slopes(self.network, inputs, truths)
        for layer_number in range(len(self.network.layers)):
            layer = self.network.layers[layer_number]
            layer_slopes = slopes[layer_number]
            # Gradient descent moves a whole array the way it moves one number: every weight at once.
            layer.weights = self.gradient_descent.step(layer.weights, layer_slopes.weights)
            layer.biases = self.gradient_descent.step(layer.biases, layer_slopes.biases)

    def loss(self, inputs: np.ndarray, truths: np.ndarray) -> float:
        """The average loss over the examples."""
        answers = self.network.answer(inputs)
        return self.backpropagation.loss.average(answers, truths)

    def train(self, inputs: np.ndarray, truths: np.ndarray, epochs: int) -> list[float]:
        """Trains for some epochs. Returns the loss before training and after each epoch."""
        losses = [self.loss(inputs, truths)]
        for epoch in range(epochs):
            self.step(inputs, truths)
            losses.append(self.loss(inputs, truths))
        return losses
