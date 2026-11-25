from pytest import approx

from neural_network.backpropagation import FiniteDifference
from neural_network.loss import SquaredError


def test_nudging_one_weight_by_0_01_gives_its_slope(mango_network, mango, show):
    slow_way = FiniteDifference(SquaredError(), nudge=0.01)
    first = mango_network.layers[0].neurons[0]
    before = slow_way.loss_of(mango_network, *mango)
    show("first weight", first.weights[0], "  loss", before)
    first.weights[0] = first.weights[0] + slow_way.nudge
    after = slow_way.loss_of(mango_network, *mango)
    show("nudged to", first.weights[0], "  loss", after, "  change", after - before)
    slope = slow_way.slope(before, after)
    show("slope = change / nudge =", slope)
    assert (before, after) == approx((0.1699, 0.1675), abs=0.0001)
    assert slope == approx(-0.24, abs=0.005)


def test_every_number_gets_a_slope_and_is_put_back(mango_network, mango, show):
    kept = [[(list(neuron.weights), neuron.bias) for neuron in layer.neurons] for layer in mango_network.layers]
    hidden, output = FiniteDifference(SquaredError()).slopes(mango_network, *mango)
    for number in range(len(hidden)):
        show("hidden neuron", number + 1, "  weight slopes", hidden[number].weights, "  bias slope", hidden[number].bias)
    show("output neuron    weight slopes", output[0].weights, "  bias slope", output[0].bias)
    assert [round(s, 2) for s in hidden[0].weights] == [-0.24, -0.12]
    assert hidden[2].weights == approx([0, 0]) and hidden[2].bias == approx(0)
    assert round(output[0].bias, 2) == -0.20
    after = [[(neuron.weights, neuron.bias) for neuron in layer.neurons] for layer in mango_network.layers]
    assert after == kept
