"""Measuring a slope: turn one number up a little, hold the other twelve still, run the pass again."""

from dataclasses import dataclass

from .network import PARTS, Example, ForwardPass, Network

FLOAT_DUST = 1e-12


def moved(before: float, after: float) -> bool:
    return abs(after - before) > FLOAT_DUST


@dataclass(frozen=True)
class Nudge:
    name: str
    size: float
    before: ForwardPass
    after: ForwardPass

    @property
    def change_in_loss(self) -> float:
        return self.after.loss - self.before.loss

    @property
    def slope(self) -> float:
        return self.change_in_loss / self.size

    @property
    def loss_moves(self) -> str:
        if not moved(self.before.loss, self.after.loss):
            return "does not move"
        return "comes down" if self.change_in_loss < 0 else "goes up"

    def road(self) -> list[str]:
        return [part for part in PARTS if moved(self.before.value(part), self.after.value(part))]


@dataclass(frozen=True)
class NudgeTest:
    size: float = 0.01

    def nudge(self, network: Network, example: Example, name: str) -> Nudge:
        nudged = Network(network.parameters.nudged(name, self.size))
        return Nudge(name, self.size, network.forward(example), nudged.forward(example))

    def nudge_each(self, network: Network, example: Example) -> dict[str, Nudge]:
        return {name: self.nudge(network, example, name) for name in network.parameters.names()}
