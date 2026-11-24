"""A loss over a model's two numbers: every spot on the map is one model with one loss."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .examples import Example
from .losses import Loss
from .models import Model, Number


def spaced(start: float, stop: float, step: float = 0.01) -> np.ndarray:
    return np.round(np.arange(start, stop + step / 2, step), 2)


@dataclass(frozen=True)
class Spot:
    model: Model
    loss: float


@dataclass(frozen=True)
class Landscape:
    loss: Loss
    model: Callable[[Number, Number], Model]
    examples: tuple[Example, ...]
    across: np.ndarray
    up: np.ndarray

    def loss_at(self, across: Number, up: Number) -> Number:
        return self.loss.average(self.model(across, up), self.examples)

    def at(self, model: Model) -> Spot:
        return Spot(model, float(self.loss_at(*model.numbers)))

    def heights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_across, grid_up = np.meshgrid(self.across, self.up)
        return grid_across, grid_up, self.loss_at(grid_across, grid_up)

    def lowest(self) -> Spot:
        grid_across, grid_up, losses = self.heights()
        row, column = np.unravel_index(losses.argmin(), losses.shape)
        return self.at(self.model(float(grid_across[row, column]), float(grid_up[row, column])))

    def row_at(self, up: float) -> "Landscape":
        return replace(self, up=np.array([up]))
