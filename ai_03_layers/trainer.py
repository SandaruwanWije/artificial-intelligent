"""Training a network with one hidden layer of ReLU neurons on the cloud."""

from dataclasses import dataclass

import numpy as np

from .cloud import SQUARE_CORNERS, Point, SeededRandom
from .network import Layer, Network
from .neuron import Neuron, relu, sigmoid

Numbers = dict[str, np.ndarray]


def random_start(hidden_neurons: int, seed: int) -> Numbers:
    random = SeededRandom(seed)

    def draw() -> float:
        return (random.uniform() * 2 - 1) * 0.9

    return {
        "hidden_weights": np.array([[draw(), draw()] for _ in range(hidden_neurons)]),
        "hidden_biases": np.array([draw() for _ in range(hidden_neurons)]),
        "output_weights": np.array([draw() for _ in range(hidden_neurons)]),
        "output_bias": np.array(0.0),
    }


def slopes(numbers: Numbers, inputs: np.ndarray, wants: np.ndarray) -> Numbers:
    hidden_sums = inputs @ numbers["hidden_weights"].T + numbers["hidden_biases"]
    hidden = relu(hidden_sums)
    answers = sigmoid(hidden @ numbers["output_weights"] + numbers["output_bias"])
    output_slope = (answers - wants) / len(wants)
    hidden_slope = np.outer(output_slope, numbers["output_weights"]) * (hidden_sums > 0)
    return {
        "hidden_weights": hidden_slope.T @ inputs,
        "hidden_biases": hidden_slope.sum(axis=0),
        "output_weights": hidden.T @ output_slope,
        "output_bias": output_slope.sum(),
    }


def as_network(numbers: Numbers) -> Network:
    hidden = Layer(tuple(
        Neuron(tuple(weights), bias, relu)
        for weights, bias in zip(numbers["hidden_weights"], numbers["hidden_biases"])
    ))
    output = Neuron(tuple(numbers["output_weights"]), float(numbers["output_bias"]), sigmoid)
    return Network((hidden, Layer((output,))))


@dataclass(frozen=True)
class Trainer:
    epochs: int = 2000
    learning_rate: float = 0.3
    momentum: float = 0.9

    def train(self, points: tuple[Point, ...], hidden_neurons: int, seed: int) -> Network:
        inputs = np.array([(point.x1, point.x2) for point in points])
        wants = np.array([point.wants for point in points], dtype=float)
        numbers = random_start(hidden_neurons, seed)
        velocity = {name: np.zeros_like(value) for name, value in numbers.items()}
        for _ in range(self.epochs):
            for name, slope in slopes(numbers, inputs, wants).items():
                velocity[name] = self.momentum * velocity[name] - self.learning_rate * slope
                numbers[name] = numbers[name] + velocity[name]
        return as_network(numbers)


@dataclass(frozen=True)
class Trial:
    hidden_neurons: int
    seed: int
    network: Network

    def guess(self, point: Point) -> int:
        return 1 if self.network.answer((point.x1, point.x2)) >= 0.5 else 0

    def wrong_points(self, points: tuple[Point, ...]) -> list[Point]:
        return [point for point in points if self.guess(point) != point.wants]

    def percent_right(self, points: tuple[Point, ...]) -> int:
        right = len(points) - len(self.wrong_points(points))
        return int(100 * right / len(points) + 0.5)

    def switched_off(self) -> list[Neuron]:
        return [neuron for neuron in self.network.hidden_neurons
                if neuron.is_switched_off(SQUARE_CORNERS)]

    def lines_switched_on(self) -> list[Neuron]:
        return [neuron for neuron in self.network.hidden_neurons
                if not neuron.is_switched_off(SQUARE_CORNERS)]
