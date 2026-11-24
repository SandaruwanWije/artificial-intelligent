"""A wide layer timed both ways: neuron by neuron in plain Python, and as one matrix."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .matrix import MatrixLayer
from .network import Layer
from .neuron import RELU, Neuron


@dataclass(frozen=True)
class RaceResult:
    neuron_by_neuron_ms: float
    one_matrix_ms: float


def best_time_ms(run: Callable[[], object], rounds: int) -> float:
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)
    return min(timings) * 1000


def random_layer(inputs: int, neurons: int, rng: np.random.Generator) -> Layer:
    return Layer("wide", tuple(
        Neuron(f"n{i}", tuple(rng.normal(size=inputs).tolist()), 0.0, RELU)
        for i in range(neurons)
    ))


def race(inputs: int = 784, neurons: int = 256, rounds: int = 5) -> RaceResult:
    rng = np.random.default_rng(0)
    layer = random_layer(inputs, neurons, rng)
    matrix = MatrixLayer.from_layer(layer)
    x = rng.normal(size=inputs)
    plain_x = x.tolist()
    return RaceResult(
        neuron_by_neuron_ms=best_time_ms(lambda: layer.fire(plain_x), rounds),
        one_matrix_ms=best_time_ms(lambda: matrix.fire(x), rounds),
    )
