"""A small network: x and y in, one hidden layer with any activation, one sigmoid output."""

import math
from dataclasses import dataclass

from .activations import Activation
from .ring import Point
from .seeded import SeededRandom


@dataclass(frozen=True)
class Hidden:
    """One hidden neuron's numbers, or how much each of them should change."""

    from_x: float
    from_y: float
    bias: float
    to_output: float

    def plus(self, other: "Hidden", times: float = 1.0) -> "Hidden":
        return Hidden(
            self.from_x + times * other.from_x,
            self.from_y + times * other.from_y,
            self.bias + times * other.bias,
            self.to_output + times * other.to_output,
        )

    def scaled(self, factor: float) -> "Hidden":
        return Hidden(factor * self.from_x, factor * self.from_y,
                      factor * self.bias, factor * self.to_output)

    def weighted_sum(self, x: float, y: float) -> float:
        return self.from_x * x + self.from_y * y + self.bias


NOTHING = Hidden(0.0, 0.0, 0.0, 0.0)


class TinyNetwork:
    def __init__(self, activation: Activation, hidden_count: int = 4, seed: int = 19):
        random = SeededRandom(seed)

        def start() -> float:
            return (random() * 2 - 1) * 0.9

        pairs = [(start(), start()) for _ in range(hidden_count)]
        biases = [start() for _ in range(hidden_count)]
        to_output = [start() for _ in range(hidden_count)]
        self.activation = activation
        self.hidden = [Hidden(wx, wy, b, w) for (wx, wy), b, w in zip(pairs, biases, to_output)]
        self.output_bias = 0.0

    def hidden_outputs(self, x: float, y: float) -> list[float]:
        return [self.activation.value(n.weighted_sum(x, y)) for n in self.hidden]

    def predict(self, x: float, y: float) -> float:
        z = self.output_bias
        for h, neuron in zip(self.hidden_outputs(x, y), self.hidden):
            z += h * neuron.to_output
        return 1 / (1 + math.exp(-z))

    def slopes(self, points: list[Point]) -> tuple[list[Hidden], float]:
        """How much the loss rises as each number grows, over all the points."""
        output_errors = [(self.predict(p.x, p.y) - p.label) / len(points) for p in points]
        hidden_slopes = []
        for neuron in self.hidden:
            from_x = from_y = bias = to_output = 0.0
            for p, error in zip(points, output_errors):
                z = neuron.weighted_sum(p.x, p.y)
                to_output += error * self.activation.value(z)
                back = error * neuron.to_output * self.activation.slope(z)
                from_x += back * p.x
                from_y += back * p.y
                bias += back
            hidden_slopes.append(Hidden(from_x, from_y, bias, to_output))
        return hidden_slopes, sum(output_errors)
