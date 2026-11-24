"""The noisy XOR cloud: four clusters of points from a seeded random source."""

import math
from dataclasses import dataclass

MASK = 0xFFFFFFFF


class SeededRandom:
    # The same 32-bit steps as the blog's figures (mulberry32), so Python draws the very same numbers
    def __init__(self, seed: int):
        self.state = seed & MASK

    def uniform(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & MASK
        t = ((self.state ^ (self.state >> 15)) * (1 | self.state)) & MASK
        t = ((t + ((t ^ (t >> 7)) * (61 | t))) & MASK) ^ t
        return ((t ^ (t >> 14)) & MASK) / 2**32

    def normal(self) -> float:
        length = math.sqrt(-2 * math.log(1 - self.uniform()))
        return length * math.cos(2 * math.pi * self.uniform())


@dataclass(frozen=True)
class Point:
    x1: float
    x2: float
    wants: int


def xor_cloud(count: int, spread: float, seed: int) -> tuple[Point, ...]:
    random = SeededRandom(seed)
    points = []
    for i in range(count):
        sign_x1 = 1 if i % 2 else -1
        sign_x2 = 1 if (i // 2) % 2 else -1
        wants = 1 if sign_x1 != sign_x2 else 0
        x1 = sign_x1 * 0.6 + random.normal() * spread
        x2 = sign_x2 * 0.6 + random.normal() * spread
        points.append(Point(x1, x2, wants))
    return tuple(points)


SQUARE = (-1.4, 1.4)
SQUARE_CORNERS = [(x1, x2) for x1 in SQUARE for x2 in SQUARE]
