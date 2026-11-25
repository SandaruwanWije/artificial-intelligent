"""Shared test pieces: the mango network, the crate of 24 mangoes, and `show`, which prints a test's steps under its name."""

import math
from random import Random

import pytest

from neural_network.activations import NO_BEND, RELU, SIGMOID
from neural_network.initialization import Initialization
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron


def build_mango_network() -> Network:
    """2 inputs, 3 ReLU neurons, 1 sigmoid answer."""
    hidden = Layer([
        Neuron([0.8, -0.4], 0.1, RELU),
        Neuron([0.3, 0.9], -0.2, RELU),
        Neuron([-0.6, 0.5], 0.3, RELU),
    ])
    output = Layer([Neuron([1.2, -0.7, 0.5], -0.1, SIGMOID)])
    return Network([hidden, output])


@pytest.fixture
def mango_network() -> Network:
    return build_mango_network()


@pytest.fixture
def new_mango_network():
    """A function that builds one more mango network, for tests that train several from the same start."""
    return build_mango_network


@pytest.fixture
def mango() -> tuple[list[float], list[float]]:
    """How yellow (1) and how soft (0.5) the mango is, and the truth: it is ripe (1)."""
    return [1.0, 0.5], [1.0]


BEST_FOR_EACH_MANGO = [
    2.62, 2.03, -0.71, 1.65, 0.26, 1.61, -0.59, 2.35, 0.69, 0.62, 1.64, 2.30,
    0.24, 1.32, 0.78, 0.62, 2.67, 0.89, -0.62, 1.37, -0.17, 0.75, 1.69, -0.01,
]


@pytest.fixture
def crate() -> list[tuple[list[float], list[float]]]:
    """24 mangoes. Each has no inputs and one truth: the answer that is right for that mango alone."""
    return [([], [best]) for best in BEST_FOR_EACH_MANGO]


@pytest.fixture
def crate_network() -> Network:
    """One number to learn: a neuron with no inputs, so its bias is its answer. It starts at -2.2."""
    return Network([Layer([Neuron([], -2.2, NO_BEND)])])


YELLOW_SOFT_RIPE = [
    (0.61, 0.66, 1), (0.06, 0.80, 0), (0.86, 0.47, 1), (0.38, 0.41, 0), (0.64, 0.66, 1), (0.09, 0.88, 0),
    (0.93, 0.80, 1), (0.81, 0.89, 1), (0.40, 0.58, 1), (0.40, 0.22, 0), (0.90, 0.93, 1), (0.13, 0.63, 0),
    (0.09, 0.58, 0), (0.56, 0.91, 1), (0.42, 0.63, 1), (0.77, 0.75, 1), (0.40, 0.65, 0), (0.89, 0.66, 1),
    (0.23, 0.57, 0), (0.68, 0.44, 0), (0.26, 0.66, 1), (0.06, 0.92, 0), (0.90, 0.15, 1), (0.52, 0.14, 0),
    (0.17, 0.75, 0), (0.93, 0.41, 1), (0.63, 0.29, 0), (0.24, 0.77, 1), (0.05, 0.87, 0), (0.09, 0.70, 0),
    (0.93, 0.93, 1), (0.39, 0.32, 0), (0.83, 0.54, 1), (0.15, 0.90, 1), (0.57, 0.57, 0), (0.57, 0.48, 0),
]


@pytest.fixture
def labelled_crate() -> list[tuple[list[float], list[float]]]:
    """36 mangoes: how yellow and how soft each one is, and the truth, ripe (1) or not (0). 18 are ripe."""
    return [([yellow, soft], [float(ripe)]) for yellow, soft, ripe in YELLOW_SOFT_RIPE]


class TimesHe(Initialization):
    """He's spread times a number: 0.5 is too small and 2 is too big. Only for tests that need a bad start."""

    def __init__(self, random: Random, times: float):
        super().__init__(random)
        self.times = times

    def spread(self, inputs: int) -> float:
        return self.times * math.sqrt(2) / math.sqrt(inputs)


@pytest.fixture
def times_he():
    """The TimesHe class, for tests that start a network with weights of the wrong size."""
    return TimesHe


def tidy(value):
    """Rounds floats to 4 places, also inside lists, so the steps are easy to read."""
    if isinstance(value, float):
        return round(value, 4) + 0.0
    if isinstance(value, list):
        return [tidy(item) for item in value]
    return value


@pytest.fixture
def show(request):
    """show("weights", [0.5, 0.5]) adds the line "weights [0.5, 0.5]" to this test's steps."""
    def add_line(*parts):
        line = " ".join(str(tidy(part)) for part in parts)
        request.node.user_properties.append(("step", line))
    return add_line


class StepPrinter:
    """After each test's name and result, prints the steps it showed, a little indented."""

    def __init__(self, terminal):
        self.terminal = terminal

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        for kind, line in report.user_properties:
            if kind == "step":
                self.terminal.write_line("      " + line)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    terminal = config.pluginmanager.getplugin("terminalreporter")
    if terminal and config.option.verbose > 0:
        config.pluginmanager.register(StepPrinter(terminal), "step_printer")
