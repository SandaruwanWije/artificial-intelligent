"""The article's examples: ten points around a line, and eight yes or no examples."""

from collections.abc import Iterator
from dataclasses import dataclass

MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class Example:
    x: float
    truth: float


def blog_random(seed: int) -> Iterator[float]:
    # The blog's own seeded source (mulberry32), so these are the very points the article draws
    state = seed
    while True:
        state = (state + 0x6D2B79F5) & MASK
        t = ((state ^ (state >> 15)) * (state | 1)) & MASK
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK) ^ t
        yield ((t ^ (t >> 14)) & MASK) / 2**32


def scatter_around_line(slope: float, crossing: float, count: int, spread: float, seed: int) -> tuple[Example, ...]:
    random = blog_random(seed)
    xs = [0.4 + i * 0.42 for i in range(count)]
    return tuple(Example(x, slope * x + crossing + (next(random) - 0.5) * spread) for x in xs)


TEN_POINTS = scatter_around_line(slope=0.8, crossing=0.5, count=10, spread=1.1, seed=4)

NO, YES = 0, 1
EIGHT_EXAMPLES = (
    Example(-2, NO), Example(-1.3, NO), Example(-0.6, NO), Example(0.4, NO),
    Example(-0.2, YES), Example(0.7, YES), Example(1.2, YES), Example(2, YES),
)
