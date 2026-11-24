"""Two weights, one loss: a long, narrow bowl, steep across and gentle along."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Weights:
    w1: float
    w2: float


@dataclass(frozen=True)
class Gradient:
    slope_for_w1: float
    slope_for_w2: float


@dataclass(frozen=True)
class Bowl:
    steepness_across: float = 6.0

    def loss(self, weights: Weights) -> float:
        return 0.5 * (weights.w1 ** 2 + self.steepness_across * weights.w2 ** 2)

    def gradient(self, weights: Weights) -> Gradient:
        return Gradient(
            slope_for_w1=weights.w1,
            slope_for_w2=self.steepness_across * weights.w2,
        )
