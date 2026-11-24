"""Counting the work in a forward pass, and how many steps two kinds of chip need for it."""

from collections.abc import Sequence
from dataclasses import dataclass

DIGIT_READER = (784, 256, 48, 10)


def multiplications_per_layer(layer_sizes: Sequence[int]) -> list[int]:
    return [inputs * neurons for inputs, neurons in zip(layer_sizes, layer_sizes[1:])]


@dataclass(frozen=True)
class Step:
    layer: int
    multiplications: int


@dataclass(frozen=True)
class Chip:
    name: str
    cores: int

    def schedule(self, per_layer: Sequence[int]) -> list[Step]:
        steps = []
        # A layer can only start once the layer before it has given its answers.
        for layer, multiplications in enumerate(per_layer):
            for start in range(0, multiplications, self.cores):
                steps.append(Step(layer, min(self.cores, multiplications - start)))
        return steps


CPU_CORE = Chip("One CPU core", cores=1)
GPU = Chip("A GPU", cores=5000)
