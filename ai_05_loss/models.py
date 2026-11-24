"""What the losses judge: a line for number answers, a rule for yes or no answers."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .examples import Example

# One spot, or a whole map of spots at once: the same code works for both
Number = float | np.ndarray


class Model(Protocol):
    @property
    def numbers(self) -> tuple[Number, Number]: ...

    def answer(self, x: float) -> Number: ...


@dataclass(frozen=True)
class Line:
    slope: Number
    crossing: Number

    @property
    def numbers(self) -> tuple[Number, Number]:
        return self.slope, self.crossing

    def answer(self, x: float) -> Number:
        return self.slope * x + self.crossing

    def miss(self, point: Example) -> Number:
        return point.truth - self.answer(point.x)

    @classmethod
    def best_fit(cls, points: Sequence[Example]) -> "Line":
        mean_x = sum(p.x for p in points) / len(points)
        mean_y = sum(p.truth for p in points) / len(points)
        rise = sum((p.x - mean_x) * (p.truth - mean_y) for p in points)
        run = sum((p.x - mean_x) ** 2 for p in points)
        slope = rise / run
        return cls(slope, mean_y - slope * mean_x)

    def __str__(self) -> str:
        return f"slope {round(self.slope, 2):g}, crossing {round(self.crossing, 2):g}"


@dataclass(frozen=True)
class Rule:
    weight: Number
    bias: Number

    @property
    def numbers(self) -> tuple[Number, Number]:
        return self.weight, self.bias

    def answer(self, x: float) -> Number:
        sureness_of_yes = 1 / (1 + np.exp(-(self.weight * x + self.bias)))
        return sureness_of_yes

    def __str__(self) -> str:
        return f"weight {round(self.weight, 2):g}, bias {round(self.bias, 2):g}"
