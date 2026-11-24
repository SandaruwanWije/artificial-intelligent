"""The blog's seeded random numbers, so Python draws the same ring and starting weights."""

import math

UINT32 = 0xFFFFFFFF


class SeededRandom:
    """Mulberry32, the generator the article's figures use."""

    def __init__(self, seed: int):
        self.state = seed & UINT32

    def __call__(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & UINT32
        a = self.state
        t = ((a ^ (a >> 15)) * (1 | a)) & UINT32
        t = ((t + ((t ^ (t >> 7)) * (61 | t))) & UINT32) ^ t
        return (t ^ (t >> 14)) / 4294967296

    def gauss(self) -> float:
        return math.sqrt(-2 * math.log(1 - self())) * math.cos(2 * math.pi * self())
