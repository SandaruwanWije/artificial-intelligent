"""RMSProp: a learning rate of its own for every weight, from the usual size of its slopes."""

import math

from neural_network.optimizers.optimizer import Optimizer


class RMSProp(Optimizer):
    """Remembers how big one weight's slopes have been, and divides the learning rate by that."""

    TINY = 1e-8  # keeps the division safe when every slope so far was 0

    def __init__(self, learning_rate: float, keep: float = 0.9):
        self.learning_rate = learning_rate
        self.keep = keep
        self.average_square = 0.0  # the running average of each slope times itself

    def step(self, weight: float, slope: float) -> float:
        square = slope * slope
        self.average_square = self.keep * self.average_square + (1 - self.keep) * square
        usual_size = math.sqrt(self.average_square)
        own_learning_rate = self.learning_rate / (usual_size + self.TINY)
        return weight - own_learning_rate * slope

    def fresh(self) -> "RMSProp":
        return RMSProp(self.learning_rate, self.keep)
