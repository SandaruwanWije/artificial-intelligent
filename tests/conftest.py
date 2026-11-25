"""Shared test pieces: the mango network, and `show`, which prints a test's steps under its name."""

import pytest

from neural_network.activations import RELU, SIGMOID
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron


@pytest.fixture
def mango_network() -> Network:
    """2 inputs, 3 ReLU neurons, 1 sigmoid answer."""
    hidden = Layer([
        Neuron([0.8, -0.4], 0.1, RELU),
        Neuron([0.3, 0.9], -0.2, RELU),
        Neuron([-0.6, 0.5], 0.3, RELU),
    ])
    output = Layer([Neuron([1.2, -0.7, 0.5], -0.1, SIGMOID)])
    return Network([hidden, output])


@pytest.fixture
def mango() -> tuple[list[float], list[float]]:
    """How yellow (1) and how soft (0.5) the mango is, and the truth: it is ripe (1)."""
    return [1.0, 0.5], [1.0]


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
