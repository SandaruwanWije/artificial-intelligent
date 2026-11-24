from dataclasses import dataclass

from .gates import Corner, LogicGate


@dataclass(frozen=True)
class Weights:
    w1: float
    w2: float
    bias: float

    def __str__(self) -> str:
        return f"w1 {self.w1:g}, w2 {self.w2:g}, bias {self.bias:g}"


class Perceptron:
    def __init__(self, weights: Weights, learning_rate: float = 0.5):
        self.weights = weights
        self.learning_rate = learning_rate

    def total(self, x1: float, x2: float) -> float:
        w = self.weights
        return w.w1 * x1 + w.w2 * x2 + w.bias

    def guess(self, x1: float, x2: float) -> int:
        return 1 if self.total(x1, x2) > 0 else 0

    def learn(self, corner: Corner) -> int:
        error = corner.answer - self.guess(corner.x1, corner.x2)
        nudge = self.learning_rate * error
        w = self.weights
        self.weights = Weights(
            w1=w.w1 + nudge * corner.x1,
            w2=w.w2 + nudge * corner.x2,
            bias=w.bias + nudge,
        )
        return error

    def wrong_corners(self, gate: LogicGate) -> list[Corner]:
        return [
            corner for corner in gate.corners
            if self.guess(corner.x1, corner.x2) != corner.answer
        ]
