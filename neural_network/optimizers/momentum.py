"""Momentum: keep most of the last step, then add the plain step."""

from neural_network.optimizers.optimizer import Optimizer


class Momentum(Optimizer):
    """Remembers one weight's last step, so steps that point the same way add up."""

    def __init__(self, learning_rate: float, keep: float = 0.9):
        self.learning_rate = learning_rate
        self.keep = keep
        self.velocity = 0.0  # the last step. No step has been taken yet.

    def step(self, weight: float, slope: float) -> float:
        plain_step = -self.learning_rate * slope
        kept = self.keep * self.velocity
        self.velocity = kept + plain_step
        return weight + self.velocity

    def fresh(self) -> "Momentum":
        return Momentum(self.learning_rate, self.keep)
