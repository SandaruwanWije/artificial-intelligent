"""Training the ring network: plain gradient descent with momentum, the same for every activation."""

import math
from dataclasses import dataclass, field

from .network import NOTHING, TinyNetwork
from .ring import Point, count_right


@dataclass
class TrainingRun:
    network: TinyNetwork
    point_count: int
    right_after_each_step: list[int] = field(default_factory=list)

    def percent_right(self, step: int = -1) -> int:
        """Worked out the way the article's chart does, so 115 of 200 shows as 57."""
        share = self.right_after_each_step[step] / self.point_count
        return math.floor(share * 100 + 0.5)


@dataclass(frozen=True)
class Trainer:
    steps: int = 2000
    learning_rate: float = 0.3
    momentum: float = 0.9

    def train(self, network: TinyNetwork, points: list[Point]) -> TrainingRun:
        run = TrainingRun(network, point_count=len(points))
        hidden_speed = [NOTHING] * len(network.hidden)
        output_bias_speed = 0.0
        for _ in range(self.steps):
            run.right_after_each_step.append(count_right(points, network.predict))
            hidden_slopes, output_bias_slope = network.slopes(points)
            output_bias_speed = (self.momentum * output_bias_speed
                                 - self.learning_rate * output_bias_slope)
            network.output_bias += output_bias_speed
            hidden_speed = [speed.scaled(self.momentum).plus(slope, -self.learning_rate)
                            for speed, slope in zip(hidden_speed, hidden_slopes)]
            network.hidden = [n.plus(speed) for n, speed in zip(network.hidden, hidden_speed)]
        run.right_after_each_step.append(count_right(points, network.predict))
        return run
