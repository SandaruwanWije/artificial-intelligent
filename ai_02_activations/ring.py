"""The ring: blue points in a small ring inside a wider ring of amber points."""

import math
from collections.abc import Callable
from dataclasses import dataclass

from .seeded import SeededRandom

INNER_RADIUS = 0.45
OUTER_RADIUS = 1.05


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    label: int


def ring(count: int = 200, noise: float = 0.1, seed: int = 5) -> list[Point]:
    random = SeededRandom(seed)
    points = []
    for i in range(count):
        label = i % 2
        angle = 2 * math.pi * random()
        radius = INNER_RADIUS if label == 1 else OUTER_RADIUS
        x = radius * math.cos(angle) + random.gauss() * noise
        y = radius * math.sin(angle) + random.gauss() * noise
        points.append(Point(x, y, label))
    return points


def count_right(points: list[Point], predict: Callable[[float, float], float]) -> int:
    return sum((1 if predict(p.x, p.y) >= 0.5 else 0) == p.label for p in points)
