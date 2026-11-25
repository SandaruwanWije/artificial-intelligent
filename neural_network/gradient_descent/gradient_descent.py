"""Gradient descent: move a number a small step against its slope."""

from neural_network.optimizers import Optimizer


class GradientDescent(Optimizer):
    """Steps a weight downhill: against its slope, by the learning rate."""

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, weight: float, slope: float) -> float:
        return weight - self.learning_rate * slope

    def fresh(self) -> "GradientDescent":
        return GradientDescent(self.learning_rate)
