"""The bump, drawn by hand with three ReLU neurons and a final neuron that adds them up."""

from dataclasses import dataclass

from .activations import ReLU

XS = (-1, 0, 1, 2, 3)


def goal(x: float) -> float:
    return max(0.0, 1 - abs(x - 1))


@dataclass(frozen=True)
class ReluNeuron:
    name: str
    weight: float
    bias: float

    def z(self, x: float) -> float:
        return self.weight * x + self.bias

    def gives(self, x: float) -> float:
        return ReLU().value(self.z(x))

    @property
    def corner(self) -> float:
        return -self.bias / self.weight


@dataclass(frozen=True)
class FinalNeuron:
    inputs: tuple[ReluNeuron, ...]
    weights: tuple[float, ...]
    bias: float = 0.0

    def y(self, x: float) -> float:
        weighted_sum = sum(w * n.gives(x) for w, n in zip(self.weights, self.inputs))
        return weighted_sum + self.bias


N1 = ReluNeuron("n1", weight=1, bias=0)
N2 = ReluNeuron("n2", weight=1, bias=-1)
N3 = ReluNeuron("n3", weight=1, bias=-2)
BUMP = FinalNeuron(inputs=(N1, N2, N3), weights=(1, -2, 1))
