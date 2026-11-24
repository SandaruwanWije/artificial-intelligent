"""The article's network, with its weights picked by hand, and the mango it looks at."""

from .network import Layer, Network
from .neuron import RELU, SIGMOID, Neuron

HIDDEN = Layer("hidden", (
    Neuron("h1", weights=(0.8, -0.4), bias=0.1, activation=RELU),
    Neuron("h2", weights=(0.3, 0.9), bias=-0.2, activation=RELU),
    Neuron("h3", weights=(-0.6, 0.5), bias=0.3, activation=RELU),
))
OUTPUT = Layer("output", (
    Neuron("y", weights=(1.2, -0.7, 0.5), bias=-0.1, activation=SIGMOID),
))
NETWORK = Network((HIDDEN, OUTPUT))

HOW_YELLOW = 1.0
HOW_SOFT = 0.5
MANGO = (HOW_YELLOW, HOW_SOFT)
RIPE = 1.0
