"""Softmax: a row of scores, one for each class, turned into chances that add up to 1."""

import numpy as np

from neural_network.arrays import ArrayActivation, ArrayLoss


class ArrayNoBend(ArrayActivation):
    """No bend: the output is the total. For a last layer that gives a raw score for each class."""

    def value(self, totals: np.ndarray) -> np.ndarray:
        return totals

    def slope(self, totals: np.ndarray) -> np.ndarray:
        return np.ones(totals.shape)


class Softmax:
    """Turns each row of scores into chances: every chance is above 0, and each row adds up to 1."""

    def chances(self, scores: np.ndarray) -> np.ndarray:
        # Taking the row's largest score from every score changes no chance, and keeps the powers of e small.
        largest = scores.max(axis=1, keepdims=True)
        raised = np.exp(scores - largest)
        row_totals = raised.sum(axis=1, keepdims=True)
        return raised / row_totals


class SoftmaxCrossEntropy(ArrayLoss):
    """Softmax, then the surprise at the chance given to the true class. Its answers are scores, not chances."""

    def __init__(self):
        self.softmax = Softmax()

    def costs(self, answers: np.ndarray, truths: np.ndarray) -> np.ndarray:
        chances = self.softmax.chances(answers)
        # A truth row is 1 at the true class and 0 elsewhere, so this keeps one chance from each row.
        chance_given_to_the_truth = (chances * truths).sum(axis=1)
        return -np.log(chance_given_to_the_truth)

    def slopes(self, answers: np.ndarray, truths: np.ndarray) -> np.ndarray:
        # Softmax and cross entropy together leave a short slope for every score: its chance minus its truth.
        return self.softmax.chances(answers) - truths


ARRAY_NO_BEND = ArrayNoBend()
