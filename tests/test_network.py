from pytest import approx

from neural_network.activations import NO_BEND, RELU
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron


def xor_network() -> Network:
    hidden = Layer([Neuron([1.0, 1.0], 0.0, RELU), Neuron([1.0, 1.0], -1.0, RELU)])
    output = Layer([Neuron([1.0, -2.0], 0.0, NO_BEND)])
    return Network([hidden, output])


def test_a_layer_gives_one_output_per_neuron(show):
    hidden = xor_network().layers[0]
    inputs = [1, 1]
    for neuron in hidden.neurons:
        show("neuron with weights", neuron.weights, "bias", neuron.bias, "-> output", neuron.output(inputs))
    show("the layer's outputs", hidden.outputs(inputs))
    assert hidden.outputs(inputs) == [2, 1]


def test_a_hidden_layer_solves_xor(show):
    network = xor_network()
    hidden, output = network.layers
    for corner, want in zip([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]):
        middle = hidden.outputs(corner)
        show(corner, "-> hidden layer", middle, "-> output layer", output.outputs(middle), "  want", want)
        assert network.answer(corner) == [want]


def test_the_mango_network_answers_0_59(mango_network, mango, show):
    inputs, _ = mango
    values = inputs
    for number, layer in enumerate(mango_network.layers, start=1):
        values = layer.outputs(values)
        show("after layer", number, values)
    assert mango_network.answer(inputs)[0] == approx(0.588, abs=0.001)
