"""Two ways to turn a miss into one number: squared error and cross entropy."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from .examples import Example
from .models import Model, Number


class Loss(ABC):
    name: str

    @abstractmethod
    def cost(self, answer: Number, truth: float) -> Number: ...

    def average(self, model: Model, examples: Sequence[Example]) -> Number:
        costs = [self.cost(model.answer(example.x), example.truth) for example in examples]
        return sum(costs) / len(costs)


class SquaredError(Loss):
    name = "squared error"

    def miss(self, answer: Number, truth: float) -> Number:
        return truth - answer

    def cost(self, answer: Number, truth: float) -> Number:
        miss = self.miss(answer, truth)
        return miss * miss


class CrossEntropy(Loss):
    name = "cross entropy"

    def cost(self, answer: Number, truth: float) -> Number:
        share_for_the_right_answer = answer if truth == 1 else 1 - answer
        surprise = np.log(1 / share_for_the_right_answer)
        return surprise
