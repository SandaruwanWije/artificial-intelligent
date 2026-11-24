"""Training: forward pass, loss, backward pass and step, round after round."""

from dataclasses import dataclass

from .backprop import Backprop
from .network import Example, ForwardPass, Network, Parameters


@dataclass(frozen=True)
class TrainingRun:
    passes: list[ForwardPass]
    end: Parameters

    @property
    def losses(self) -> list[float]:
        return [passed.loss for passed in self.passes]

    @property
    def answers(self) -> list[float]:
        return [passed.y for passed in self.passes]


@dataclass(frozen=True)
class Trainer:
    learning_rate: float = 0.5
    rounds: int = 40

    def step(self, network: Network, example: Example) -> Network:
        slopes = Backprop(network, example).sweep().slopes
        old = network.parameters
        new = {name: old.value(name) - self.learning_rate * slopes[name] for name in old.names()}
        return Network(Parameters(**new))

    def train(self, network: Network, example: Example) -> TrainingRun:
        passes = [network.forward(example)]
        for _ in range(self.rounds):
            network = self.step(network, example)
            passes.append(network.forward(example))
        return TrainingRun(passes, network.parameters)
