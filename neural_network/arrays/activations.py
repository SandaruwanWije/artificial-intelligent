"""Activations for arrays: every total of a layer is bent in one step."""

from abc import ABC, abstractmethod

import numpy as np


class ArrayActivation(ABC):
    """Turns a whole array of totals into outputs, and says how steep the bend is at each total."""

    @abstractmethod
    def value(self, totals: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def slope(self, totals: np.ndarray) -> np.ndarray: ...


class ArrayReLU(ArrayActivation):
    """0 for every total below 0; the total itself above 0."""

    def value(self, totals: np.ndarray) -> np.ndarray:
        return np.maximum(0, totals)

    def slope(self, totals: np.ndarray) -> np.ndarray:
        above_zero = totals > 0
        # True counts as 1 and False as 0, so this is 1 where the total is above 0.
        return above_zero * 1.0


class ArraySigmoid(ArrayActivation):
    """A smooth step from 0 to 1."""

    def value(self, totals: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-totals))

    def slope(self, totals: np.ndarray) -> np.ndarray:
        s = self.value(totals)
        return s * (1 - s)


ARRAY_RELU, ARRAY_SIGMOID = ArrayReLU(), ArraySigmoid()
