"""Losses: one number for how wrong the answers are. 0 means perfect."""

import math
from abc import ABC, abstractmethod


class Loss(ABC):
    """Compares an answer with the truth, and says how the loss moves when the answer moves."""

    @abstractmethod
    def cost(self, answer: float, truth: float) -> float: ...

    @abstractmethod
    def slope(self, answer: float, truth: float) -> float: ...

    def total(self, answers: list[float], truths: list[float]) -> float:
        total = 0.0
        for i in range(len(answers)):
            total = total + self.cost(answers[i], truths[i])
        return total


class SquaredError(Loss):
    """The miss times itself, so big misses cost far more than small ones."""

    def cost(self, answer: float, truth: float) -> float:
        miss = answer - truth
        return miss * miss

    def slope(self, answer: float, truth: float) -> float:
        miss = answer - truth
        return 2 * miss


class CrossEntropy(Loss):
    """The surprise at the truth, for answers that are a chance between 0 and 1."""

    def cost(self, answer: float, truth: float) -> float:
        if truth == 1:
            chance_given_to_the_truth = answer
        else:
            chance_given_to_the_truth = 1 - answer
        return -math.log(chance_given_to_the_truth)

    def slope(self, answer: float, truth: float) -> float:
        return (answer - truth) / (answer * (1 - answer))
