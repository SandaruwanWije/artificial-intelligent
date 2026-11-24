from dataclasses import dataclass, field

from .gates import Corner, LogicGate
from .perceptron import Perceptron, Weights


@dataclass(frozen=True)
class Fix:
    pass_number: int
    corner: Corner
    guess: int
    weights_after: Weights


@dataclass
class TrainingRun:
    start: Weights
    after_each_pass: list[Weights] = field(default_factory=list)
    mistakes_per_pass: list[int] = field(default_factory=list)
    fixes: list[Fix] = field(default_factory=list)

    @property
    def passes(self) -> int:
        return len(self.mistakes_per_pass)

    @property
    def learned(self) -> bool:
        return self.passes > 0 and self.mistakes_per_pass[-1] == 0


@dataclass(frozen=True)
class Trainer:
    max_passes: int = 8

    def train(self, perceptron: Perceptron, gate: LogicGate) -> TrainingRun:
        run = TrainingRun(start=perceptron.weights)
        for pass_number in range(1, self.max_passes + 1):
            mistakes = 0
            for corner in gate.corners:
                guess = perceptron.guess(corner.x1, corner.x2)
                if perceptron.learn(corner) != 0:
                    mistakes += 1
                    run.fixes.append(Fix(pass_number, corner, guess, perceptron.weights))
            run.mistakes_per_pass.append(mistakes)
            run.after_each_pass.append(perceptron.weights)
            if mistakes == 0:
                break
        return run
