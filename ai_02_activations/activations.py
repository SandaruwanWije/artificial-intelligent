"""The activation function: the bend at the end of every neuron."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence


ALMOST_FLAT = 0.05


class Activation(ABC):
    name: str

    @abstractmethod
    def value(self, z: float) -> float: ...

    @abstractmethod
    def slope(self, z: float) -> float: ...

    def is_almost_flat(self, z: float) -> bool:
        return abs(self.slope(z)) < ALMOST_FLAT

    def lowest(self, zs: Sequence[float]) -> tuple[float, float]:
        lowest_z = min(zs, key=self.value)
        return lowest_z, self.value(lowest_z)


class Step(Activation):
    name = "Step"

    def value(self, z: float) -> float:
        return 1.0 if z > 0 else 0.0

    def slope(self, z: float) -> float:
        no_slope_at_the_jump = math.nan
        return no_slope_at_the_jump if z == 0 else 0.0


class Sigmoid(Activation):
    name = "Sigmoid"

    def value(self, z: float) -> float:
        return 1 / (1 + math.exp(-z))

    def slope(self, z: float) -> float:
        return self.value(z) * (1 - self.value(z))


class Tanh(Activation):
    name = "Tanh"

    def value(self, z: float) -> float:
        return math.tanh(z)

    def slope(self, z: float) -> float:
        return 1 - math.tanh(z) ** 2


class ReLU(Activation):
    name = "ReLU"

    def value(self, z: float) -> float:
        return max(0.0, z)

    def slope(self, z: float) -> float:
        return 1.0 if z > 0 else 0.0


class GELU(Activation):
    """The tanh form of GELU, the one GPT and BERT use."""

    name = "GELU"
    SCALE = math.sqrt(2 / math.pi)
    CUBE = 0.044715

    def _inner(self, z: float) -> float:
        return self.SCALE * (z + self.CUBE * z**3)

    def value(self, z: float) -> float:
        return 0.5 * z * (1 + math.tanh(self._inner(z)))

    def slope(self, z: float) -> float:
        t = math.tanh(self._inner(z))
        inner_slope = self.SCALE * (1 + 3 * self.CUBE * z * z)
        return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * inner_slope


class NoBend(Activation):
    name = "No bend"

    def value(self, z: float) -> float:
        return z

    def slope(self, z: float) -> float:
        return 1.0


FOUR_CURVES: tuple[Activation, ...] = (Sigmoid(), Tanh(), ReLU(), GELU())
