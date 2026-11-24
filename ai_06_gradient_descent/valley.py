"""One weight, one loss: a curve shaped like a valley."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Valley:
    lowest_weight: float = 1.0
    lowest_loss: float = 0.3
    steepness: float = 0.35

    def loss(self, weight: float) -> float:
        return self.steepness * (weight - self.lowest_weight) ** 2 + self.lowest_loss

    def slope(self, weight: float) -> float:
        return 2 * self.steepness * (weight - self.lowest_weight)
