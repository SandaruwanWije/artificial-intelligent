"""Article 7's network: two inputs, three ReLU neurons, one sigmoid answer."""

import math
from dataclasses import dataclass, fields, replace

HIDDEN = ("h1", "h2", "h3")
PARTS = ("h1 sum", "h1", "h2 sum", "h2", "h3 sum", "h3", "y sum", "y", "loss")
WIRE_INTO_Y = {"h1": "w7", "h2": "w8", "h3": "w9"}


@dataclass(frozen=True)
class Example:
    x1: float
    x2: float
    right_answer: float


@dataclass(frozen=True)
class Parameters:
    w1: float
    w2: float
    b1: float
    w3: float
    w4: float
    b2: float
    w5: float
    w6: float
    b3: float
    w7: float
    w8: float
    w9: float
    b4: float

    def names(self) -> list[str]:
        return [field.name for field in fields(self)]

    def value(self, name: str) -> float:
        return getattr(self, name)

    def nudged(self, name: str, by: float) -> "Parameters":
        return replace(self, **{name: self.value(name) + by})


@dataclass(frozen=True)
class Feed:
    neuron: str
    multiplied_by: str | None


FEEDS = {
    "w1": Feed("h1", "x1"), "w2": Feed("h1", "x2"), "b1": Feed("h1", None),
    "w3": Feed("h2", "x1"), "w4": Feed("h2", "x2"), "b2": Feed("h2", None),
    "w5": Feed("h3", "x1"), "w6": Feed("h3", "x2"), "b3": Feed("h3", None),
    "w7": Feed("y", "h1"), "w8": Feed("y", "h2"), "w9": Feed("y", "h3"), "b4": Feed("y", None),
}


@dataclass(frozen=True)
class ForwardPass:
    x1: float
    x2: float
    h1_sum: float
    h2_sum: float
    h3_sum: float
    h1: float
    h2: float
    h3: float
    y_sum: float
    y: float
    loss: float

    def value(self, part: str) -> float:
        return getattr(self, part.replace(" ", "_"))


def relu(weighted_sum: float) -> float:
    return max(0.0, weighted_sum)


def sigmoid(weighted_sum: float) -> float:
    return 1 / (1 + math.exp(-weighted_sum))


class Network:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def forward(self, example: Example) -> ForwardPass:
        p, x1, x2 = self.parameters, example.x1, example.x2
        h1_sum = p.w1 * x1 + p.w2 * x2 + p.b1
        h2_sum = p.w3 * x1 + p.w4 * x2 + p.b2
        h3_sum = p.w5 * x1 + p.w6 * x2 + p.b3
        h1, h2, h3 = relu(h1_sum), relu(h2_sum), relu(h3_sum)
        y_sum = p.w7 * h1 + p.w8 * h2 + p.w9 * h3 + p.b4
        y = sigmoid(y_sum)
        loss = (y - example.right_answer) ** 2
        return ForwardPass(x1, x2, h1_sum, h2_sum, h3_sum, h1, h2, h3, y_sum, y, loss)
