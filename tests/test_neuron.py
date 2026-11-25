from pytest import approx

from neural_network.activations import NO_BEND, RELU, SIGMOID
from neural_network.neuron import Neuron


def test_the_total_is_each_input_times_its_weight_plus_the_bias(show):
    neuron = Neuron(weights=[0.8, -0.4], bias=0.1)
    inputs = [1.0, 0.5]
    for x, weight in zip(inputs, neuron.weights):
        show("input", x, "x weight", weight, "=", x * weight)
    show("add them up, plus the bias", neuron.bias, "-> total", neuron.total(inputs))
    assert neuron.total(inputs) == approx(0.7)


def test_the_activation_bends_the_total(show):
    inputs = [1.0, 0.5]
    neuron = Neuron([-0.6, 0.5], 0.3)
    total = neuron.total(inputs)
    outputs = []
    for activation in (NO_BEND, RELU, SIGMOID):
        neuron.activation = activation
        outputs.append(neuron.output(inputs))
        show(type(activation).__name__, "turns the total", total, "into the output", outputs[-1])
    assert outputs == approx([-0.05, 0, 0.4875], abs=0.0001)


def test_relu_is_the_default(show):
    neuron = Neuron([1], 0)
    show("no activation given, so the neuron uses", type(neuron.activation).__name__)
    assert neuron.activation is RELU
