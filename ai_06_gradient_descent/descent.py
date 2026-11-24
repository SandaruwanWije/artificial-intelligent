"""Gradient descent: feel the slope, step the other way, repeat."""

from dataclasses import dataclass

from .bowl import Bowl, Weights
from .valley import Valley


@dataclass(frozen=True)
class Step:
    weight: float
    slope: float
    size: float


@dataclass(frozen=True)
class ValleyDescent:
    learning_rate: float
    steps: tuple[Step, ...]
    end: float


@dataclass(frozen=True)
class BowlDescent:
    learning_rate: float
    path: tuple[Weights, ...]
    losses: tuple[float, ...]

    @property
    def lowest_loss_step(self) -> int:
        return self.losses.index(min(self.losses))


@dataclass(frozen=True)
class GradientDescent:
    learning_rate: float

    def step(self, weight: float, slope: float) -> float:
        return weight - self.learning_rate * slope

    def down_the_valley(self, valley: Valley, start: float, steps: int) -> ValleyDescent:
        weight, taken = start, []
        for _ in range(steps):
            slope = valley.slope(weight)
            next_weight = self.step(weight, slope)
            taken.append(Step(weight, slope, size=next_weight - weight))
            weight = next_weight
        return ValleyDescent(self.learning_rate, tuple(taken), end=weight)

    def across_the_bowl(self, bowl: Bowl, start: Weights, steps: int) -> BowlDescent:
        path = [start]
        for _ in range(steps):
            here = path[-1]
            gradient = bowl.gradient(here)
            # Both slopes are read before either weight moves, so they move at once.
            path.append(Weights(
                w1=self.step(here.w1, gradient.slope_for_w1),
                w2=self.step(here.w2, gradient.slope_for_w2),
            ))
        return BowlDescent(self.learning_rate, tuple(path), tuple(map(bowl.loss, path)))
