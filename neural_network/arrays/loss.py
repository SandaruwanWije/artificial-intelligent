"""Losses for arrays: how wrong every answer is, worked out in one step."""

from abc import ABC, abstractmethod

import numpy as np


class ArrayLoss(ABC):
    """Compares every answer with its truth, and says how the loss moves when each answer moves."""

    @abstractmethod
    def costs(self, answers: np.ndarray, truths: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def slopes(self, answers: np.ndarray, truths: np.ndarray) -> np.ndarray: ...

    def average(self, answers: np.ndarray, truths: np.ndarray) -> float:
        """The loss of one example, on average."""
        examples = answers.shape[0]
        return float(self.costs(answers, truths).sum() / examples)


class ArrayCrossEntropy(ArrayLoss):
    """The surprise at each truth, for answers that are a chance between 0 and 1."""

    def costs(self, answers: np.ndarray, truths: np.ndarray) -> np.ndarray:
        chance_given_to_the_truth = np.where(truths == 1, answers, 1 - answers)
        return -np.log(chance_given_to_the_truth)

    def slopes(self, answers: np.ndarray, truths: np.ndarray) -> np.ndarray:
        return (answers - truths) / (answers * (1 - answers))
