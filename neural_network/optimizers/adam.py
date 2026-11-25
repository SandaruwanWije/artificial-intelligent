"""Adam: the memory of Momentum and the learning rates of RMSProp, in one rule."""

import math

from neural_network.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    """Remembers one weight's average slope and the usual size of its slopes, and steps by one over the other."""

    TINY = 1e-8  # keeps the division safe when every slope so far was 0

    def __init__(self, learning_rate: float, keep_slope: float = 0.9, keep_square: float = 0.999):
        self.learning_rate = learning_rate
        self.keep_slope = keep_slope
        self.keep_square = keep_square
        self.average_slope = 0.0  # the running average of the slopes
        self.average_square = 0.0  # the running average of each slope times itself
        # How much of each memory is still its empty start. 1 is all of it.
        self.slope_empty = 1.0
        self.square_empty = 1.0

    def step(self, weight: float, slope: float) -> float:
        square = slope * slope
        self.average_slope = self.keep_slope * self.average_slope + (1 - self.keep_slope) * slope
        self.average_square = self.keep_square * self.average_square + (1 - self.keep_square) * square
        # Both memories start at 0, so they read too small. Divide each by how full it is.
        self.slope_empty = self.slope_empty * self.keep_slope
        self.square_empty = self.square_empty * self.keep_square
        full_slope = self.average_slope / (1 - self.slope_empty)
        full_square = self.average_square / (1 - self.square_empty)
        usual_size = math.sqrt(full_square)
        return weight - self.learning_rate * full_slope / (usual_size + self.TINY)

    def fresh(self) -> "Adam":
        return Adam(self.learning_rate, self.keep_slope, self.keep_square)
